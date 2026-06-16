from utils import Registry, check_availability

DATASET_REGISTRY = Registry("DATASET")


def build_dataset(cfg):
    """
        Builds a list of datasets, each corresponding to a domain.
    """
    available_datasets = DATASET_REGISTRY.registered_names()

    source_domains = cfg.DATASET.SOURCE_DOMAINS
    target_domains = cfg.DATASET.TARGET_DOMAINS

    if not source_domains or not target_domains:
        raise ValueError("Source and target domains are required")
        
    for domain in target_domains:
        if domain not in source_domains:
            raise ValueError(f"Domain {domain} is not in the source domains")

    list_of_datasets = []
    # build target domains which is a subset of source domains
    for index, domain_name in enumerate(target_domains):
        check_availability(domain_name, available_datasets)

        # Create individual dataset instance
        dataset = DATASET_REGISTRY.get(domain_name)(cfg, domain_label=index, verbose=True)
        list_of_datasets.append(dataset)


    return list_of_datasets
