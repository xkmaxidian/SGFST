import copy
import math
from typing import Literal, Optional

import decoupler as dc
import numpy as np
import omnipath as op
import pandas as pd
from anndata import AnnData


def extract_gp_dict_from_omnipath_lr_interactions(
        species: Literal["mouse", "human"],
        min_curation_effort: int = 2,
        load_from_disk: bool = False,
        save_to_disk: bool = False,
        lr_network_file_path: Optional[str] = "../data/gene_programs/"
                                              "omnipath_lr_network.csv",
        gene_orthologs_mapping_file_path: Optional[str] = "../data/gene_"
                                                          "annotations/human_"
                                                          "mouse_gene_orthologs.csv") -> dict:
    """
    Retrieve 724 human ligand-receptor interactions from OmniPath and extract
    them into a gene program dictionary. OmniPath is a database of molecular
    biology prior knowledge that combines intercellular communication data from
    many different resources (all resources for intercellular communication
    included in OmniPath can be queried via
    ´op.requests.Intercell.resources()´). If ´species´ is ´mouse´, orthologs
    from human interactions are returned.

    Parts of the implementation are inspired by
    https://workflows.omnipathdb.org/intercell-networks-py.html (01.10.2022).

    Parameters
    ----------
    species:
        Species for which the gene programs will be extracted. The default is
        human. Human genes are mapped to mouse orthologs using a mapping file.
        NicheCompass contains a default mapping file stored under
        "<root>/data/gene_annotations/human_mouse_gene_orthologs.csv", which was
        created with Ensembl BioMart
        (http://www.ensembl.org/info/data/biomart/index.html).
    min_curation_effort:
        Indicates how many times an interaction has to be described in a
        paper and mentioned in a database to be included in the retrieval.
    load_from_disk:
        If ´True´, the OmniPath ligand receptor interactions will be loaded from
        disk instead of from the OmniPath library.
    save_to_disk:
        If ´True´, the OmniPath ligand receptor interactions will additionally
        be stored on disk. Only applies if ´load_from_disk´ is ´False´.
    lr_network_file_path:
        Path of the file where the OmniPath ligand receptor interactions will be
        stored (if ´save_to_disk´ is ´True´) or loaded from (if ´load_from_disk´
        is ´True´).
    gene_orthologs_mapping_file_path:
        Path of the file where the gene orthologs mapping is stored if species
        is ´mouse´.
    plot_gp_gene_count_distributions:
        If ´True´, display the distribution of gene programs per number of
        source and target genes.
    gp_gene_count_distributions_save_path:
        Path of the file where the gene program gene count distribution plot
        will be saved if ´plot_gp_gene_count_distributions´ is ´True´.

    Returns
    ----------
    gp_dict:
        Nested dictionary containing the OmniPath ligand-receptor interaction
        gene programs with keys being gene program names and values being
        dictionaries with keys ´sources´, ´targets´, ´sources_categories´, and
        ´targets_categories´, where ´sources´ contains the OmniPath ligands,
        ´targets´ contains the OmniPath receptors, ´sources_categories´ contains
        the categories of the sources, and ´targets_categories´ contains
        the categories of the targets.
    """
    if not load_from_disk:
        # Define intercell_network categories to be retrieved (see
        # https://workflows.omnipathdb.org/intercell-networks-py.html,
        # https://omnipath.readthedocs.io/en/latest/api/omnipath.interactions.import_intercell_network.html#omnipath.interactions.import_intercell_network)
        intercell_df = op.interactions.import_intercell_network(
            include=["omnipath", "pathwayextra", "ligrecextra"])
        lr_interaction_df = intercell_df[
            (intercell_df["category_intercell_source"] == "ligand")
            & (intercell_df["category_intercell_target"] == "receptor")]
        if save_to_disk:
            lr_interaction_df.to_csv(lr_network_file_path, index=False)
    else:
        lr_interaction_df = pd.read_csv(lr_network_file_path, index_col=0)

    # Only keep curated interactions (see
    # https://r.omnipathdb.org/reference/filter_intercell_network.html)
    lr_interaction_df = lr_interaction_df[
        lr_interaction_df["curation_effort"] >= min_curation_effort]

    # Group receptors by ligands
    grouped_lr_interaction_df = lr_interaction_df.groupby(
        "genesymbol_intercell_source")["genesymbol_intercell_target"].agg(
        list).reset_index()

    # Resolve protein complexes into individual genes
    def _is_na(x):
        return x is None or (isinstance(x, float) and math.isnan(x))

    def compute_elementwise_func(lst, func):
        if _is_na(lst):
            seq = []
        elif isinstance(lst, (list, tuple, set)):
            seq = list(lst)
        else:
            seq = [lst]
        return [func(item) for item in seq]

    def resolve_protein_complexes(x):
        if _is_na(x):
            return []
        if not isinstance(x, str):
            x = str(x)
        if x.startswith("COMPLEX:"):
            body = x[len("COMPLEX:"):]
            return [p for p in body.split("_") if p]
        return [x]

    grouped_lr_interaction_df["sources"] = grouped_lr_interaction_df[
        "genesymbol_intercell_source"].apply(
        lambda x: list(set(resolve_protein_complexes(x))))
    grouped_lr_interaction_df["sources_categories"] = grouped_lr_interaction_df[
        "sources"].apply(lambda x: ["ligand"] * len(x))
    grouped_lr_interaction_df["targets"] = grouped_lr_interaction_df[
        "genesymbol_intercell_target"].apply(
        lambda x: list(set([element for sublist in compute_elementwise_func(x, resolve_protein_complexes) for element in
                            sublist])))
    grouped_lr_interaction_df["targets_categories"] = grouped_lr_interaction_df[
        "targets"].apply(lambda x: ["receptor"] * len(x))

    # Extract gene programs and store in nested dict
    gp_dict = {}
    for _, row in grouped_lr_interaction_df.iterrows():
        gp_dict[row["genesymbol_intercell_source"] +
                "_ligand_receptor_GP"] = {
            "sources": row["sources"],
            "targets": row["targets"],
            "sources_categories": row["sources_categories"],
            "targets_categories": row["targets_categories"]}

    if species == "mouse":
        # Create mapping df to map from human genes to mouse orthologs
        mapping_df = pd.read_csv(gene_orthologs_mapping_file_path)
        grouped_mapping_df = mapping_df.groupby(
            "Gene name")["Mouse gene name"].agg(list).reset_index()

        # Map all genes in the gene programs to their orthologs from the mapping
        # df or capitalize them if no orthologs are found (one human gene can
        # have multiple mouse orthologs)
        for _, gp in gp_dict.items():
            gp["sources"] = [element for nested_list_l1 in [
                list_element for nested_list_l2 in [
                    grouped_mapping_df[
                        grouped_mapping_df["Gene name"] == source][
                        "Mouse gene name"].values.tolist() if
                    source in grouped_mapping_df["Gene name"].values else
                    [[source.capitalize()]] for source in gp["sources"]]
                for list_element in nested_list_l2]
                             for element in nested_list_l1]
            gp["targets"] = [element for nested_list_l1 in [
                list_element for nested_list_l2 in [
                    grouped_mapping_df[
                        grouped_mapping_df["Gene name"] == target][
                        "Mouse gene name"].values.tolist() if
                    target in grouped_mapping_df["Gene name"].values else
                    [[target.capitalize()]] for target in gp["targets"]]
                for list_element in nested_list_l2]
                             for element in nested_list_l1]
            gp["sources_categories"] = ["ligand"] * len(gp["sources"])
            gp["targets_categories"] = ["receptor"] * len(gp["targets"])

    return gp_dict
