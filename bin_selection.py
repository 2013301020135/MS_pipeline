#!/usr/bin/env python
"""Analyze dynesty results, extract posteriors and evidences, draw plots for bin selection.
    Written by Yang Liu (liuyang@shao.ac.cn)."""

import pickle
import corner
import os
import shutil
import argparse
import copy
# import glob
import subprocess
import re
import time
# import math
import logging
import traceback
import sys
# import inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
from matplotlib.table import Table
from pathlib import Path
from multiprocessing import Process, Queue
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
# from datetime import datetime
from optparse import OptionParser
from dynesty import plotting


class PulsarNoiseModelPipeline:
    def __init__(self, **kwargs):
        self.kargs = kwargs
        self.logger = None
        self.setup_logging()
        self.models_to_run = self.parse_models()
        self.total_steps = len(self.models_to_run) * 2  # Each model has 2 steps
        self.current_progress = 0

    def setup_logging(self):
        """Set up log files."""
        log_file = f"{self.kargs['datadir']}/pipeline_{self.kargs['psrname']}.log"
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                            handlers=[logging.FileHandler(log_file, mode='a'), logging.StreamHandler()])
        self.logger = logging.getLogger(__name__)

    def parse_models(self):
        """Read the models list to be tested

        :return: models: the models to be tested in model selection
        """
        models = self.kargs['modelname']
        valid_models = ["RN+DM", "RN+DM+SV", "RN+DM+SW", "RN+DM+SV+SW"]
        invalid_models = [m for m in models if m not in valid_models]
        if invalid_models:
            self.logger.error(f"Name of invalid models: {invalid_models}")
            self.logger.error(f"Name of valid models: {valid_models}")
            raise ValueError(f"Invalid model names: {invalid_models}!")
        self.logger.info(f"Models to be tested: {models}")
        return models

    def run_command(self, command, output_file=None, **kwargs):
        """Run commands with subprocess and write log files"""
        self.logger.info(f"Execute commands: {' '.join(command)}")
        try:
            if output_file:
                with open(output_file, 'w') as outfile:
                    result = subprocess.run(command, check=True, stdout=outfile, stderr=subprocess.STDOUT, text=True)
            else:
                result = subprocess.run(command, check=True, capture_output=True, text=True)
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Command execution failed: {e}")
            if output_file and os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    self.logger.error(f"Error output: {f.read()}")
            return False

    def generate_commented_parfile(self):
        """Generate the parameters file after commenting out parameters."""
        par_file = f"{self.kargs['datadir']}/{self.kargs['psrname']}-posttn.par"
        command = ["python", f"{self.kargs['workdir']}/comment_posttn.py", "-p", par_file]
        return self.run_command(command)

    def generate_noisefile(self):
        """Generate the noisefiles for pulsar from parameters file."""
        par_file = f"{self.kargs['datadir']}/{self.kargs['psrname']}-posttn.par"
        os.makedirs(f"{self.kargs['datadir']}/noisefiles/", exist_ok=True)
        command = ["python", f"{self.kargs['workdir']}/make_json.py", "-p", par_file, "-o", "noisefiles/"]
        return self.run_command(command)

    def get_model_components(self, model_name):
        """Get the list of components in models.

        :return: list_comps: the components list in models
        """
        components = {"RN": {"idx": None, "name": "RN", "bin_param": "RN_bin", "posterior_key": "rn_k_dropbin"},
                      "DM": {"idx": 2, "name": "DM", "bin_param": "DM_bin", "posterior_key": "dm_gp_k_dropbin"},
                      "SV": {"idx": 4, "name": "SV", "bin_param": "SV_bin", "posterior_key": "sv_gp_k_dropbin"},
                      "SW": {"idx": None, "name": "SW", "bin_param": "SW_bin", "posterior_key": "sw_gp_k_dropbin"}}
        model_parts = model_name.split('+')
        list_comps = [components[part] for part in model_parts if part in components]
        return list_comps

    def get_bin_number(self, comp_name, step, bin_numbers=None):
        """Get the bin number of given components."""
        if step == 1:
            if comp_name == "RN":
                return self.kargs['RN_bin']
            elif comp_name == "DM":
                return self.kargs['DM_bin']
            elif comp_name == "SV":
                return self.kargs['SV_bin']
            elif comp_name == "SW":
                return self.kargs['SW_bin']
        else:
            if bin_numbers and comp_name in bin_numbers:
                return bin_numbers[comp_name]
            else:
                return getattr(self.kargs, f"{comp_name}_bin")

    def build_model_string(self, components, step, bin_numbers=None):
        """Build the string for model in fitting.

        :return: model_string: the string describing the model
        """
        model_parts = ["TM/WN,fix,ecorr"]
        for comp in components:
            bin_num = self.get_bin_number(comp["name"], step, bin_numbers)
            if comp["name"] == "RN":
                if step == 1:
                    model_parts.append(f"RN,nb={bin_num},dropbin")
                else:
                    model_parts.append(f"RN,nb={bin_num}")
            elif comp["name"] == "DM":
                if step == 1:
                    model_parts.append(f"RN,idx=2,nb={bin_num},dropbin")
                else:
                    model_parts.append(f"RN,idx=2,nb={bin_num}")
            elif comp["name"] == "SV":
                if step == 1:
                    model_parts.append(f"RN,idx=4,nb={bin_num},dropbin")
                else:
                    model_parts.append(f"RN,idx=4,nb={bin_num}")
            elif comp["name"] == "SW":  # SW ???
                if step == 1:
                    model_parts.append(f"SW,nb={bin_num},dropbin")
                else:
                    model_parts.append(f"SW,nb={bin_num}")
        model_string = "/".join(model_parts)
        return model_string

    def extract_bin_numbers(self, output_dir, model_name):
        """Extract the bin component number from the results of step 1 run.

        :return: bin_comps: the fix bin component number determined by the step 1 run results
        """
        suffix_parts = []
        components = self.get_model_components(model_name)
        for comp in components:
            bin_num = self.get_bin_number(comp["name"], 1)
            suffix_parts.append(f"{comp['name']}b{bin_num}")
        model_suffix = "+".join(suffix_parts)
        posterior_file = f"{output_dir}/{self.kargs['psrname']}_{model_suffix}_posterior.txt"
        self.logger.info(f"Try extracting bin numbers from: {posterior_file}")
        if not os.path.exists(posterior_file):
            self.logger.warning(f"Posterior file {posterior_file} does not exist!")
            default_bins = {"RN": self.kargs['RN_bin'], "DM": self.kargs['DM_bin'],
                            "SV": self.kargs['SV_bin'], "SW": self.kargs['SW_bin']}
            return {k: v for k, v in default_bins.items() if k in model_name}
        try:
            df = pd.read_csv(posterior_file, sep='\s+', header=0)
            df.columns = [col.strip() for col in df.columns]
            bin_comps = {}
            for comp in components:
                matching_rows = df[df['Parameters'].str.contains(comp["posterior_key"])]
                if not matching_rows.empty:
                    mean_value = matching_rows['mean'].values[0]
                    rounded_bin = int(np.ceil(mean_value))
                    bin_comps[comp["name"]] = rounded_bin
                    self.logger.info(f"Set the bin number of {comp['name']} to {rounded_bin} based on {mean_value}")
                else:
                    self.logger.warning(f"Warning: the parameter {comp['posterior_key']} corresponds to bin number of "
                                        f"{comp['name']} not found!!! ")
                    bin_comps[comp["name"]] = getattr(self.kargs, f"{comp['name']}_bin")
            return bin_comps
        except Exception as e:
            self.logger.error(f"Extract bin number failed: {e}")
            default_bins = {"RN": self.kargs['RN_bin'], "DM": self.kargs['DM_bin'],
                            "SV": self.kargs['SV_bin'], "SW": self.kargs['SW_bin']}
            return {k: v for k, v in default_bins.items() if k in model_name}

    def run_enterprise_analysis(self, model_name, step, bin_numbers=None):
        """Run Bayesian analysis with enterprise.

        :param model_name: the name of the noise model
        :param step: the step for the model, 1 for bin selection, 2 for fixed bin numbers
        :param bin_numbers: the number of bins used for noise model
        """
        if self.kargs['replot']:
            return True
        output_dir = f"{self.kargs['datadir']}/{model_name}/"
        par_file = f"{self.kargs['datadir']}/{self.kargs['psrname']}_all.par"
        tim_file = f"{self.kargs['datadir']}/{self.kargs['psrname']}_all.tim"
        os.makedirs(output_dir, exist_ok=True)
        components = self.get_model_components(model_name)
        model_string = self.build_model_string(components, step, bin_numbers)
        sampler_config = f"dynesty,nlive={self.kargs['numlive']},Nthread={self.kargs['thread']}"
        command = ["python", f"{self.kargs['entdir']}/enterprise_bayesian_analysis.py", "-o", output_dir,
                   "-p", par_file, "-t", tim_file, "--noisedirs", f"{self.kargs['datadir']}/noisefiles/",
                   "--maxobs", str(self.kargs['max_obs']), "--sampler", sampler_config, "-m", model_string]
        output_file = f"{output_dir}/step{step}_analysis_output.txt"
        return self.run_command(command, output_file)

    def run_plot_results(self, model_name, step, bin_numbers=None):
        """Run plotting scripts and extract results.

        :param model_name: the name of the noise model
        :param step: the step for the model, 1 for bin selection, 2 for fixed bin numbers
        :param bin_numbers: the number of bins used for noise model
        """
        output_dir = f"{self.kargs['datadir']}/{model_name}/"
        if step == 1:
            suffix_parts = []
            components = self.get_model_components(model_name)
            for comp in components:
                bin_num = self.get_bin_number(comp["name"], step)
                suffix_parts.append(f"{comp['name']}b{bin_num}")
            model_suffix = "+".join(suffix_parts)
            command = ["python", f"{self.kargs['workdir']}/plot_dynesty_results.py", "-o", output_dir,
                       "-p", self.kargs['psrname'], "-m", model_suffix, "-b", str(self.kargs['burn'])]
            if self.kargs['replot']:
                command.extend(["--re"])
        else:
            if bin_numbers:
                suffix_parts = []
                components = self.get_model_components(model_name)
                for comp in components:
                    if comp["name"] in bin_numbers:
                        suffix_parts.append(f"{comp['name']}{bin_numbers[comp['name']]}")
                    else:
                        bin_num = self.get_bin_number(comp["name"], step)
                        suffix_parts.append(f"{comp['name']}{bin_num}")
                model_suffix = "+".join(suffix_parts)
            else:
                model_suffix = "final"
            command = ["python", f"{self.kargs['workdir']}/plot_dynesty_results.py", "-o", output_dir,
                       "-p", self.kargs['psrname'], "-m", model_suffix, "-b", str(self.kargs['burn']), "-e"]
            if self.kargs['replot']:
                command.extend(["--re"])
        output_file = f"{output_dir}/step{step}_plot_output.txt"
        return self.run_command(command, output_file)

    def process_single_model(self, model_name, progress_queue):
        """Process the complete progress of given model.

        :param model_name: the name of the noise model
        :param progress_queue: the queue of progress
        """
        self.logger.info(f"Begin processing model: {model_name}")
        self.logger.info(f"Model {model_name}: Step 1 - bin number selection")
        if self.run_enterprise_analysis(model_name, 1):
            if self.run_plot_results(model_name, 1):
                progress_queue.put(1)
        bin_numbers = self.extract_bin_numbers(f"{self.kargs['datadir']}/{model_name}/", model_name)
        self.logger.info(f"The bin numbers of model {model_name} are: {bin_numbers}")
        self.logger.info(f"Model {model_name}: Step 2 - model selection with fixed bin number")
        if self.run_enterprise_analysis(model_name, 2, bin_numbers):
            if self.run_plot_results(model_name, 2, bin_numbers):
                progress_queue.put(1)
        self.logger.info(f"Complete processing model: {model_name}")

    def run_pipeline(self):
        """Run the complete pipeline."""
        self.logger.info("Begin pipeline for advanced noise model selection of single source")
        self.logger.info(f"Source name: PSR {self.kargs['psrname']}")
        self.logger.info(f"List of models to be tested: {self.models_to_run}")
        self.logger.info(f"Maximum RN bins allowed: {self.kargs['RN_bin']}")
        self.logger.info(f"Maximum DM bins allowed: {self.kargs['DM_bin']}")
        self.logger.info(f"Maximum SV bins allowed: {self.kargs['SV_bin']}")
        self.logger.info(f"Maximum SW bins allowed: {self.kargs['SW_bin']}")
        self.logger.info(f"Number of live points: {self.kargs['numlive']}")
        self.logger.info(f"Number of threads: {self.kargs['thread']}")
        self.logger.info(f"Number of maximum observation allowed: {self.kargs['max_obs']}")
        if self.kargs['replot']:
            self.logger.info("Replot: Skip Stage 1 and Stage 2")
        else:
            self.logger.info("Stage 1: Commenting parameter file")
            if not self.generate_commented_parfile():
                self.logger.error("Parameter file commenting failed")
                return False
            self.logger.info("Stage 2: Generating noise file")
            if not self.generate_noisefile():
                self.logger.error("Noise file generation failed")
                return False
        self.logger.info("Stage 3: Fitting all models in parallel")
        processes = []
        progress_queue = Queue()
        pbar = tqdm(total=self.total_steps, desc="Overall progress for model selection")
        for model in self.models_to_run:
            p = Process(target=self.process_single_model, args=(model, progress_queue))
            p.start()
            processes.append(p)
        completed_steps = 0
        while completed_steps < self.total_steps:
            try:
                progress_queue.get(timeout=None)
                completed_steps += 1
                pbar.update(1)
                pbar.set_description(f"Overall progress ({completed_steps}/{self.total_steps})")
            except KeyboardInterrupt:
                self.logger.info("Pipeline interrupted by user")
                break
            except Exception as e:
                self.logger.error(f"Unexpected error in progress monitoring: {e}")
                if all(not p.is_alive() for p in processes):
                    break
        pbar.close()
        for p in processes:
            p.join()
        self.logger.info("Stage 4: Performing model selection")
        self.perform_model_selection()
        self.logger.info("Stage 5: Generating summary PDF")
        self.generate_summary_pdf()
        self.logger.info("Pipeline execution completed!")
        return True

    def perform_model_selection(self):
        """Perform model selection using data from evidence files and calculate Bayes factors."""
        bin_evidence_file = f"{self.kargs['datadir']}/{self.kargs['psrname']}_bin_evidences.csv"
        final_evidence_file = f"{self.kargs['datadir']}/{self.kargs['psrname']}_evidences.csv"
        if os.path.exists(bin_evidence_file):
            try:
                bin_evidence_df = pd.read_csv(bin_evidence_file)
                if 'log evidence' in bin_evidence_df.columns and 'Model' in bin_evidence_df.columns:
                    best_bin_df = bin_evidence_df.loc[bin_evidence_df.groupby('Model')['log evidence'].idxmax()]
                    best_bin_evidence_df = best_bin_df.sort_values('log evidence', ascending=False)
                    self.logger.info("Bin selection evidence comparison (best configuration for each model):")
                    self.logger.info(f"\n{best_bin_evidence_df[['Model', 'log evidence']].to_string(index=False)}")
                else:
                    self.logger.warning("Required columns not found in bin evidence file")
            except Exception as e:
                self.logger.error(f"Failed to process bin evidence file: {e}")
        else:
            self.logger.warning(f"Bin evidence file not found: {bin_evidence_file}")
        if os.path.exists(final_evidence_file):
            try:
                final_evidence_df = pd.read_csv(final_evidence_file)
                if 'log evidence' in final_evidence_df.columns and 'Model' in final_evidence_df.columns:
                    best_final_df = final_evidence_df.loc[final_evidence_df.groupby('Model')['log evidence'].idxmax()]
                    best_final_evidence_df = best_final_df.sort_values('log evidence', ascending=False)
                    best_model_row = best_final_evidence_df.iloc[0]
                    best_model = best_model_row['Model']
                    best_evidence = best_model_row['log evidence']
                    self.logger.info("Final model evidence comparison (best configuration for each model):")
                    comparison_data = []
                    for idx, row in best_final_evidence_df.iterrows():
                        model_name = row['Model']
                        evidence = row['log evidence']
                        evidence_diff = evidence - best_evidence
                        if model_name == best_model:
                            evidence_diff_str = "0.0 (best)"
                            bayes_factor_str = "1 (reference)"
                        else:
                            bayes_factor = np.exp(best_evidence - evidence)
                            evidence_diff_str = f"{evidence_diff:.3f} (relative to best)"
                            bayes_factor_str = f"{bayes_factor:.3f} (best/this)"
                        comparison_data.append({'Model': model_name, 'log evidence': f"{evidence:.3f}",
                                                'log Z diff': evidence_diff_str, 'Bayes factor': bayes_factor_str})
                    comparison_df = pd.DataFrame(comparison_data)
                    self.logger.info(f"\n{comparison_df.to_string(index=False)}")
                    self.logger.info(f"\nBest model: {best_model}")
                    self.logger.info(f"Best model evidence (log): {best_evidence:.3f}")
                    self.logger.info("Best model bin configuration:")
                    if 'RN bin' in best_model_row and pd.notna(best_model_row['RN bin']):
                        self.logger.info(f"  RN: {int(best_model_row['RN bin'])} bins")
                    if 'DM bin' in best_model_row and pd.notna(best_model_row['DM bin']):
                        self.logger.info(f"  DM: {int(best_model_row['DM bin'])} bins")
                    if 'SV bin' in best_model_row and pd.notna(best_model_row['SV bin']):
                        self.logger.info(f"  SV: {int(best_model_row['SV bin'])} bins")
                    if 'SW bin' in best_model_row and pd.notna(best_model_row['SW bin']):
                        self.logger.info(f"  SW: {int(best_model_row['SW bin'])} bins")
                else:
                    self.logger.error("Required columns not found in final evidence file")
            except Exception as e:
                self.logger.error(f"Failed to process final evidence file: {e}")
        else:
            self.logger.warning(f"Final evidence file not found: {final_evidence_file}")

    def generate_summary_pdf(self):
        """Generate a summary PDF file with corner plots and result tables"""
        pdf_path = f"{self.kargs['datadir']}/{self.kargs['psrname']}_summary.pdf"
        try:
            with pdf_backend.PdfPages(pdf_path) as pdf:
                self.logger.info(f"Generating summary PDF: {pdf_path}")
                self._add_bin_selection_corner_plots(pdf)
                self._add_bin_selection_table(pdf)
                self._add_model_selection_corner_plots(pdf)
                self._add_model_selection_table(pdf)
            self.logger.info(f"Successfully generated summary PDF: {pdf_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to generate summary PDF: {e}")
            return False

    def _add_bin_selection_corner_plots(self, pdf):
        """Add bin selection corner plots to PDF."""
        all_possible_models = ["RN+DM", "RN+DM+SV", "RN+DM+SW", "RN+DM+SV+SW"]
        bin_evidence_file = f"{self.kargs['datadir']}/{self.kargs['psrname']}_bin_evidences.csv"
        best_configurations = {}
        if os.path.exists(bin_evidence_file):
            try:
                bin_evidence_df = pd.read_csv(bin_evidence_file)
                if 'log evidence' in bin_evidence_df.columns and 'Model' in bin_evidence_df.columns:
                    best_bin_df = bin_evidence_df.loc[bin_evidence_df.groupby('Model')['log evidence'].idxmax()]
                    for _, row in best_bin_df.iterrows():
                        model = row['Model']
                        bins_config = {}
                        for comp in ['RN', 'DM', 'SV', 'SW']:
                            bin_col = f'{comp} bin'
                            if bin_col in row and pd.notna(row[bin_col]):
                                bins_config[comp] = int(row[bin_col])
                        best_configurations[model] = bins_config
            except Exception as e:
                self.logger.warning(f"Could not read bin evidence file to get best configurations: {e}")
        for model in all_possible_models:
            suffix_parts = []
            components = self.get_model_components(model)
            if model in best_configurations:
                bins_config = best_configurations[model]
                for comp in components:
                    comp_name = comp['name']
                    if comp_name in bins_config:
                        bin_num = bins_config[comp_name]
                    else:
                        bin_num = self.get_bin_number(comp_name, 1)
                    suffix_parts.append(f"{comp_name}b{bin_num}")
            model_suffix = "+".join(suffix_parts)
            corner_plot_path = f"{self.kargs['datadir']}/{model}/cornerplot_{self.kargs['psrname']}_{model_suffix}.png"
            if os.path.exists(corner_plot_path):
                try:
                    img = plt.imread(corner_plot_path)
                    height, width = img.shape[0], img.shape[1]
                    max_dim = 10
                    if width > height:
                        fig_width = max_dim
                        fig_height = max_dim * height / width
                    else:
                        fig_height = max_dim
                        fig_width = max_dim * width / height
                    fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)
                    ax = fig.add_axes([0, 0, 1, 0.95])
                    ax.imshow(img, aspect='auto')
                    ax.axis('off')
                    plt.suptitle(f"Bin Selection - {model}", fontsize=16, y=0.98)
                    pdf.savefig(fig, bbox_inches='tight', pad_inches=0.1, dpi=300)
                    plt.close(fig)
                except Exception as e:
                    self.logger.warning(f"Could not add corner plot {corner_plot_path}: {e}")
                    fig = plt.figure(figsize=(12, 12))
                    plt.text(0.5, 0.5, f"Corner plot not available for {model}\n{model_suffix}",
                             ha='center', va='center', fontsize=16)
                    plt.title(f"Bin Selection - {model}", fontsize=16)
                    plt.axis('off')
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)

    def _add_bin_selection_table(self, pdf):
        """Add bin selection results table to PDF with new structure."""
        bin_evidence_file = f"{self.kargs['datadir']}/{self.kargs['psrname']}_bin_evidences.csv"
        if not os.path.exists(bin_evidence_file):
            fig = plt.figure(figsize=(12, 12))
            plt.text(0.5, 0.5, f"Bin selection evidence data not available", ha='center', va='center', fontsize=16)
            plt.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            return
        try:
            df = pd.read_csv(bin_evidence_file)
            best_df = df.loc[df.groupby('Model')['log evidence'].idxmax()]
            df = best_df.sort_values('log evidence', ascending=False)
            best_evidence = df.iloc[0]['log evidence']
            fig = plt.figure(figsize=(12, 12))  # Landscape orientation
            fig.suptitle(f"{self.kargs['psrname']} - Bin Selection Results Summary", fontsize=16, fontweight='bold')
            models = df['Model'].tolist()
            table_data = []
            evidence_row = ['log evidence ± std']
            for _, row in df.iterrows():
                evidence_str = f"{row['log evidence']:.3f} ± {row['log evidence std']:.3f}"
                evidence_row.append(evidence_str)
            table_data.append(evidence_row)
            delta_evidence_row = ['log Z diff']
            for _, row in df.iterrows():
                delta_evidence = row['log evidence'] - best_evidence
                delta_evidence_row.append(f"{delta_evidence:.3f}")
            table_data.append(delta_evidence_row)
            components = ['RN', 'DM', 'SV', 'SW']
            for comp in components:
                if f'{comp} Ncoeff mean' in df.columns and f'{comp} Ncoeff std' in df.columns:
                    ncoeff_row = [f'{comp} Ncoeff']
                    for _, row in df.iterrows():
                        if pd.notna(row[f'{comp} Ncoeff mean']) and pd.notna(row[f'{comp} Ncoeff std']):
                            ncoeff_str = f"{row[f'{comp} Ncoeff mean']:.1f} ± {row[f'{comp} Ncoeff std']:.1f}"
                        else:
                            ncoeff_str = '--'
                        ncoeff_row.append(ncoeff_str)
                    table_data.append(ncoeff_row)
                bin_col = f'{comp} bin'
                if bin_col in best_df.columns:
                    bin_row = [f'{comp} bins']
                    for _, row in best_df.iterrows():
                        if pd.notna(row[bin_col]):
                            bin_str = f"{int(row[bin_col])}"
                        else:
                            bin_str = '--'
                        bin_row.append(bin_str)
                    table_data.append(bin_row)
            ax = plt.subplot(111)
            ax.axis('off')
            headers = ['Parameter'] + models
            table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center', 
                             bbox=[0, 0, 1, 0.95])
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(2, 1)
            for i in range(len(table_data) + 1):
                if i > 0:
                    table[(i, 1)].set_facecolor('#E8F5E8')
                    table[(i, 1)].set_text_props(weight='bold')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            self.logger.error(f"Failed to create bin selection table: {e}")
            self.logger.error(traceback.format_exc())

    def _add_model_selection_corner_plots(self, pdf):
        """Add model selection corner plots to PDF."""
        all_possible_models = ["RN+DM", "RN+DM+SV", "RN+DM+SW", "RN+DM+SV+SW"]
        final_evidence_file = f"{self.kargs['datadir']}/{self.kargs['psrname']}_evidences.csv"
        best_configurations = {}
        if os.path.exists(final_evidence_file):
            try:
                final_evidence_df = pd.read_csv(final_evidence_file)
                if 'log evidence' in final_evidence_df.columns and 'Model' in final_evidence_df.columns:
                    best_final_df = final_evidence_df.loc[final_evidence_df.groupby('Model')['log evidence'].idxmax()]
                    for _, row in best_final_df.iterrows():
                        model = row['Model']
                        bins_config = {}
                        for comp in ['RN', 'DM', 'SV', 'SW']:
                            bin_col = f'{comp} bin'
                            if bin_col in row and pd.notna(row[bin_col]):
                                bins_config[comp] = int(row[bin_col])
                        best_configurations[model] = bins_config
            except Exception as e:
                self.logger.warning(f"Could not read final evidence file to get best configurations: {e}")
        for model in all_possible_models:
            model_dir = f"{self.kargs['datadir']}/{model}/"
            corner_plot_path = None
            if model in best_configurations:
                bins_config = best_configurations[model]
                suffix_parts = []
                components = self.get_model_components(model)
                for comp in components:
                    comp_name = comp['name']
                    if comp_name in bins_config:
                        bin_num = bins_config[comp_name]
                    else:
                        bin_num = self.get_bin_number(comp_name, 1)
                    suffix_parts.append(f"{comp_name}{bin_num}")
                model_suffix = "+".join(suffix_parts)
                corner_plot_path = f"{model_dir}/cornerplot_{self.kargs['psrname']}_{model_suffix}.png"
            #corner_plots = [f for f in os.listdir(model_dir) if f.startswith(f"cornerplot_{self.kargs['psrname']}_")]
            #fixed_bin_plots = [f for f in corner_plots if not any(f"{comp}b" in f for comp in ['RN', 'DM', 'SV', 'SW'])]
            #if fixed_bin_plots:
            #    corner_plot_path = f"{model_dir}/{fixed_bin_plots[0]}"
            #else:
            #    corner_plot_path = f"{model_dir}/{corner_plots[0]}" if corner_plots else None
            if corner_plot_path and os.path.exists(corner_plot_path):
                try:
                    img = plt.imread(corner_plot_path)
                    height, width = img.shape[0], img.shape[1]
                    max_dim = 10
                    if width > height:
                        fig_width = max_dim
                        fig_height = max_dim * height / width
                    else:
                        fig_height = max_dim
                        fig_width = max_dim * width / height
                    fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)
                    ax = fig.add_axes([0, 0, 1, 0.95])
                    ax.imshow(img, aspect='auto')
                    ax.axis('off')
                    plt.suptitle(f"Model Selection - {model}", fontsize=16, y=0.98)
                    pdf.savefig(fig, bbox_inches='tight', pad_inches=0.1, dpi=300)
                    plt.close(fig)
                except Exception as e:
                    self.logger.warning(f"Could not add corner plot {corner_plot_path}: {e}")
                    fig = plt.figure(figsize=(12, 12))
                    plt.text(0.5, 0.5, f"Corner plot not available for {model}", ha='center', va='center', fontsize=16)
                    plt.title(f"Model Selection - {model}", fontsize=16)
                    plt.axis('off')
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)

    def _add_model_selection_table(self, pdf):
        """Add model selection results table to PDF with new structure."""
        final_evidence_file = f"{self.kargs['datadir']}/{self.kargs['psrname']}_evidences.csv"
        if not os.path.exists(final_evidence_file):
            fig = plt.figure(figsize=(12, 12))
            plt.text(0.5, 0.5, f"Model selection evidence data not available", ha='center', va='center', fontsize=16)
            plt.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            return
        try:
            df = pd.read_csv(final_evidence_file)
            best_df = df.loc[df.groupby('Model')['log evidence'].idxmax()]
            df = best_df.sort_values('log evidence', ascending=False)
            best_evidence = df.iloc[0]['log evidence']
            fig = plt.figure(figsize=(12, 12))  # Landscape orientation
            fig.suptitle(f"{self.kargs['psrname']} - Model Selection Results Summary", fontsize=16, fontweight='bold')
            models = df['Model'].tolist()
            table_data = []
            evidence_row = ['log evidence ± std']
            for _, row in df.iterrows():
                evidence_str = f"{row['log evidence']:.3f} ± {row['log evidence std']:.3f}"
                evidence_row.append(evidence_str)
            table_data.append(evidence_row)
            delta_evidence_row = ['log Z diff']
            for _, row in df.iterrows():
                if row['log evidence'] == best_evidence:
                    delta_evidence = "0 (best)"
                else:
                    delta_evidence = f"{(row['log evidence'] - best_evidence):.3f}"
                delta_evidence_row.append(delta_evidence)
            table_data.append(delta_evidence_row)
            bayes_factor_row = ['Bayes factor']
            for _, row in df.iterrows():
                if row['log evidence'] == best_evidence:
                    bayes_factor = "1 (best)"
                else:
                    bayes_factor = f"{np.exp(best_evidence - row['log evidence']):.3f}"
                bayes_factor_row.append(bayes_factor)
            table_data.append(bayes_factor_row)
            components = ['RN', 'DM', 'SV', 'SW']
            for comp in components:
                if f"{comp} log10amp mean" in df.columns and f"{comp} log10amp std" in df.columns:
                    log10amp_row = [f"{comp} log10amp"]
                    for _, row in df.iterrows():
                        if pd.notna(row[f"{comp} log10amp mean"]) and pd.notna(row[f"{comp} log10amp std"]):
                            log10amp_str = f"{row[f'{comp} log10amp mean']:.3f} ± {row[f'{comp} log10amp std']:.3f}"
                        else:
                            log10amp_str = '--'
                        log10amp_row.append(log10amp_str)
                    table_data.append(log10amp_row)
                if f"{comp} gamma mean" in df.columns and f"{comp} gamma std" in df.columns:
                    gamma_row = [f"{comp} gamma"]
                    for _, row in df.iterrows():
                        if pd.notna(row[f"{comp} gamma mean"]) and pd.notna(row[f"{comp} gamma std"]):
                            gamma_str = f"{row[f'{comp} gamma mean']:.3f} ± {row[f'{comp} gamma std']:.3f}"
                        else:
                            gamma_str = '--'
                        gamma_row.append(gamma_str)
                    table_data.append(gamma_row)
                if f"{comp} bin" in df.columns:
                    bin_row = [f"{comp} bin"]
                    for _, row in df.iterrows():
                        if pd.notna(row[f"{comp} bin"]):
                            bin_str = f"{int(row[f'{comp} bin'])}"
                        else:
                            bin_str = '--'
                        bin_row.append(bin_str)
                    table_data.append(bin_row)
            ax = plt.subplot(111)
            ax.axis('off')
            headers = ['Parameter'] + models
            table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center',
                             bbox=[0, 0, 1, 0.95])
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(2, 1)
            for i in range(len(table_data) + 1):
                if i > 0:
                    table[(i, 1)].set_facecolor('#E8F5E8')
                    table[(i, 1)].set_text_props(weight='bold')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            self.logger.error(f"Failed to create model selection table: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The pipeline for analyze dynesty results, "
                                                 "extract posteriors and evidences, draw plots for bin selection",
                                     usage="usage: %prog  [options]")
    parser.add_argument('--workd', '--work-dir', type=str, dest="workdir",
                        default='/cluster/home/liuyang/Pulsar/MS_pipeline/',
                        help='The absolute path of this bin selection package')
    parser.add_argument('--entd', '--enterprise-dir', type=str, dest="entdir",
                        default='/cluster/home/liuyang/Software/noise_modeling/',
                        help='The absolute path of enterprise package')
    parser.add_argument("-d", "--datadir", type=str, dest="datadir", default=".",
                        help="Path to the data directory of the pulsar.")
    parser.add_argument("-p", "--psrname", type=str, dest="psrname", required=True,
                        help="Name of the pulsar.")
    parser.add_argument("-m", "--modelname", type=str, dest="modelname", nargs="+", required=True,
                        choices=["RN+DM", "RN+DM+SV", "RN+DM+SW", "RN+DM+SV+SW"],
                        help="Name of the timing model to be tested")
    parser.add_argument("-b", "--burn", type=float, dest='burn', default=0.3,
                        help="Fraction of chains to be burned.")
    parser.add_argument("--rnb", "--RN-bin", type=int, dest='RN_bin', default=100,
                        help="The maximum number of bins allowed for RN.")
    parser.add_argument("--dmb", "--DM-bin", type=int, dest='DM_bin', default=100,
                        help="The maximum number of bins allowed for DM.")
    parser.add_argument("--svb", "--SV-bin", type=int, dest='SV_bin', default=100,
                        help="The maximum number of bins allowed for SV.")
    parser.add_argument("--swb", "--SW-bin", type=int, dest='SW_bin', default=100,
                        help="The maximum number of bins allowed for SW.")
    parser.add_argument("-n", "--num-live", type=int, dest='numlive', default=2000,
                        help="The number of live points used in dynesty runs.")
    parser.add_argument("-t", "--thread", type=int, dest='thread', default=32,
                        help="The number of thread used in dynesty runs.")
    parser.add_argument("--mobs", "--max-obs", type=int, dest='max_obs', default=100000,
                        help="The maximum number of observations allowed.")
    parser.add_argument("--re", "--replot", action="store_true", dest="replot", default=False, 
                        help="If True: Replot and regenerate summary pdf with existing fitting results;"
                        "If False: Execute the whole bin selection pipeline normally.")

    args = parser.parse_args()
    os.makedirs(args.datadir, exist_ok=True)
    args_dict = vars(args)
    pipeline = PulsarNoiseModelPipeline(**args_dict)
    success = pipeline.run_pipeline()
