def print_conf(config: dict):
    print("Simulating BCI data")
    print(f"\tregressor: \t\t\t{config['regressor']}")
    print(f"\tacquisition: \t\t\t{config['acquisition']}")
    print(f"\tinitializer: \t\t\t{config['initializer']}")
    print(f"\treplicator: \t\t\t{config['replicator']}")
    print(f"\tselector: \t\t\t{config['selector']}")
    print(f"\tconvergence measure: \t\t{config['convergence_measure']}")
    print(f"\tdimensionality: \t\t{config['dimension']}")
    print(f"\trandom samples: \t\t{config['random_sample_size']}")
    print(f"\tinformed samples: \t\t{config['informed_sample_size']}")
    print(
        f"\texperiment: \t\t\t{config['experiment']}; participant: "
        f"{config['participant']}; condition: {config['condition']}"
    )