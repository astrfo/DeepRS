def create_hyperparameter_list(result_dir_path, param):
    with open(result_dir_path + 'hyperparameter_list.txt', mode='w', encoding='utf-8') as f:
        param_element = ",\n".join(f"  {k}: {v}" for k, v in param.items())
        f.write("param: {\n")
        f.write(f"{param_element}\n")
        f.write("}\n")
