#!/usr/bin/env python
"""Generate summary PDF from existing results for pulsar noise model selection.
    Written by Yang Liu (liuyang@shao.ac.cn)."""

import os
import argparse
import logging
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend


class PulsarNoiseModelPDFGenerator:
    def __init__(self, **kwargs):
        self.kargs = kwargs
        self.logger = None
        self.setup_logging()

    def setup_logging(self):
        """Set up logging."""
        log_file = f"{self.kargs['datadir']}/pdf_generator_{self.kargs['psrname']}.log"
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                            handlers=[logging.FileHandler(log_file, mode='a'), logging.StreamHandler()])
        self.logger = logging.getLogger(__name__)

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

    def generate_summary_pdf(self):
        """Generate a summary PDF file with corner plots and result tables from existing results"""
        pdf_path = f"{self.kargs['datadir']}/{self.kargs['psrname']}_summary.pdf"
        try:
            with pdf_backend.PdfPages(pdf_path) as pdf:
                self.logger.info(f"Generating summary PDF from existing results: {pdf_path}")
                self._add_bin_selection_corner_plots(pdf)
                self._add_bin_selection_table(pdf)
                self._add_model_selection_corner_plots(pdf)
                self._add_model_selection_table(pdf)
            self.logger.info(f"Successfully generated summary PDF: {pdf_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to generate summary PDF: {e}")
            self.logger.error(traceback.format_exc())
            return False

    def _add_bin_selection_corner_plots(self, pdf):
        """Add bin selection corner plots to PDF for specified models."""
        for model in self.kargs['modelname']:
            bin_evidence_file = f"{self.kargs['datadir']}/{self.kargs['psrname']}_bin_evidences.csv"
            best_bins = {}
            if os.path.exists(bin_evidence_file):
                try:
                    bin_evidence_df = pd.read_csv(bin_evidence_file)
                    if 'log evidence' in bin_evidence_df.columns and 'Model' in bin_evidence_df.columns:
                        model_df = bin_evidence_df[bin_evidence_df['Model'] == model]
                        if not model_df.empty:
                            best_row = model_df.loc[model_df['log evidence'].idxmax()]
                            for comp in ['RN', 'DM', 'SV', 'SW']:
                                bin_col = f'{comp} bin'
                                if bin_col in best_row and pd.notna(best_row[bin_col]):
                                    best_bins[comp] = int(best_row[bin_col])
                except Exception as e:
                    self.logger.warning(f"Could not read bin evidence file for {model}: {e}")
            suffix_parts = []
            components = self.get_model_components(model)
            for comp in components:
                comp_name = comp['name']
                if comp_name in best_bins:
                    bin_num = best_bins[comp_name]
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
                    plt.text(0.5, 0.5, f"Corner plot not available for {model}", ha='center', va='center', fontsize=16)
                    plt.title(f"Bin Selection - {model}", fontsize=16)
                    plt.axis('off')
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)

    def _add_bin_selection_table(self, pdf):
        """Add bin selection results table to PDF."""
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
            df = df[df['Model'].isin(self.kargs['modelname'])]
            best_indices = df.groupby('Model')['log evidence'].idxmax()
            best_df = df.loc[best_indices]
            best_df = best_df.sort_values('log evidence', ascending=False)
            best_evidence = best_df.iloc[0]['log evidence']
            fig = plt.figure(figsize=(12, 12))
            fig.suptitle(f"{self.kargs['psrname']} - Bin Selection Results Summary", fontsize=16, fontweight='bold')
            models = best_df['Model'].tolist()
            table_data = []
            evidence_row = ['log evidence ± std']
            for _, row in best_df.iterrows():
                evidence_str = f"{row['log evidence']:.3f} ± {row['log evidence std']:.3f}"
                evidence_row.append(evidence_str)
            table_data.append(evidence_row)
            delta_evidence_row = ['Delta log Z']
            for _, row in best_df.iterrows():
                delta_evidence = row['log evidence'] - best_evidence
                delta_evidence_row.append(f"{delta_evidence:.3f}")
            table_data.append(delta_evidence_row)
            components = ['RN', 'DM', 'SV', 'SW']
            for comp in components:
                if f'{comp} Ncoeff mean' in best_df.columns and f'{comp} Ncoeff std' in best_df.columns:
                    ncoeff_row = [f'{comp} Ncoeff']
                    for _, row in best_df.iterrows():
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
        """Add model selection corner plots to PDF for specified models."""
        for model in self.kargs['modelname']:
            final_evidence_file = f"{self.kargs['datadir']}/{self.kargs['psrname']}_evidences.csv"
            best_bins = {}
            if os.path.exists(final_evidence_file):
                try:
                    final_evidence_df = pd.read_csv(final_evidence_file)
                    if 'log evidence' in final_evidence_df.columns and 'Model' in final_evidence_df.columns:
                        model_df = final_evidence_df[final_evidence_df['Model'] == model]
                        if not model_df.empty:
                            best_row = model_df.loc[model_df['log evidence'].idxmax()]
                            for comp in ['RN', 'DM', 'SV', 'SW']:
                                bin_col = f'{comp} bin'
                                if bin_col in best_row and pd.notna(best_row[bin_col]):
                                    best_bins[comp] = int(best_row[bin_col])
                except Exception as e:
                    self.logger.warning(f"Could not read final evidence file for {model}: {e}")
            model_dir = f"{self.kargs['datadir']}/{model}/"
            corner_plot_path = None
            if best_bins:
                suffix_parts = []
                components = self.get_model_components(model)
                for comp in components:
                    comp_name = comp['name']
                    if comp_name in best_bins:
                        suffix_parts.append(f"{comp_name}{best_bins[comp_name]}")
                model_suffix = "+".join(suffix_parts)
                corner_plot_path = f"{model_dir}/cornerplot_{self.kargs['psrname']}_{model_suffix}.png"
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
        """Add model selection results table to PDF."""
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
            df = df[df['Model'].isin(self.kargs['modelname'])]
            best_indices = df.groupby('Model')['log evidence'].idxmax()
            best_df = df.loc[best_indices]
            best_df = best_df.sort_values('log evidence', ascending=False)
            best_evidence = best_df.iloc[0]['log evidence']
            fig = plt.figure(figsize=(12, 12))  # Landscape orientation
            fig.suptitle(f"{self.kargs['psrname']} - Model Selection Results Summary", fontsize=16, fontweight='bold')
            models = best_df['Model'].tolist()
            table_data = []
            evidence_row = ['log evidence ± std']
            for _, row in best_df.iterrows():
                evidence_str = f"{row['log evidence']:.3f} ± {row['log evidence std']:.3f}"
                evidence_row.append(evidence_str)
            table_data.append(evidence_row)
            delta_evidence_row = ['Delta log Z']
            for _, row in best_df.iterrows():
                if row['log evidence'] == best_evidence:
                    delta_evidence = "0 (best)"
                else:
                    delta_evidence = f"{(row['log evidence'] - best_evidence):.3f}"
                delta_evidence_row.append(delta_evidence)
            table_data.append(delta_evidence_row)
            bayes_factor_row = ['Bayes factor']
            for _, row in best_df.iterrows():
                if row['log evidence'] == best_evidence:
                    bayes_factor = "1 (best)"
                else:
                    bayes_factor = f"{np.exp(best_evidence - row['log evidence']):.3f}"
                bayes_factor_row.append(bayes_factor)
            table_data.append(bayes_factor_row)
            components = ['RN', 'DM', 'SV', 'SW']
            for comp in components:
                if f"{comp} log10amp mean" in best_df.columns and f"{comp} log10amp std" in best_df.columns:
                    log10amp_row = [f"{comp} log10amp"]
                    for _, row in best_df.iterrows():
                        if pd.notna(row[f"{comp} log10amp mean"]) and pd.notna(row[f"{comp} log10amp std"]):
                            log10amp_str = f"{row[f'{comp} log10amp mean']:.3f} ± {row[f'{comp} log10amp std']:.3f}"
                        else:
                            log10amp_str = '--'
                        log10amp_row.append(log10amp_str)
                    table_data.append(log10amp_row)
                if f"{comp} gamma mean" in best_df.columns and f"{comp} gamma std" in best_df.columns:
                    gamma_row = [f"{comp} gamma"]
                    for _, row in best_df.iterrows():
                        if pd.notna(row[f"{comp} gamma mean"]) and pd.notna(row[f"{comp} gamma std"]):
                            gamma_str = f"{row[f'{comp} gamma mean']:.3f} ± {row[f'{comp} gamma std']:.3f}"
                        else:
                            gamma_str = '--'
                        gamma_row.append(gamma_str)
                    table_data.append(gamma_row)
                if f"{comp} bin" in best_df.columns:
                    bin_row = [f"{comp} bin"]
                    for _, row in best_df.iterrows():
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
    parser = argparse.ArgumentParser(description="Generate summary PDF from existing dynesty results "
                                                 "for pulsar noise model selection",
                                     usage="usage: %prog [options]")
    parser.add_argument("-d", "--datadir", type=str, dest="datadir", default=".",
                        help="Path to the data directory of the pulsar.")
    parser.add_argument("-p", "--psrname", type=str, dest="psrname", required=True,
                        help="Name of the pulsar.")
    parser.add_argument("-m", "--modelname", type=str, dest="modelname", nargs="+", required=True,
                        choices=["RN+DM", "RN+DM+SV", "RN+DM+SW", "RN+DM+SV+SW"],
                        help="Name of the timing model to be included in the summary")
    args = parser.parse_args()
    args_dict = vars(args)
    pdf_generator = PulsarNoiseModelPDFGenerator(**args_dict)
    print(f"Generating summary PDF for pulsar {args.psrname}:")
    print(f"Included models: {', '.join(args.modelname)}")
    success = pdf_generator.generate_summary_pdf()
