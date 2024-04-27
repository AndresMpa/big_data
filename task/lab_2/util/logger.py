from util.env_vars import config


def trace_log(timestamp, log):
    if (config["verbose"]):
        print("Índices de reemplazo:")
        for column, index_dict in log.items():
            print(column)
            for index, value in index_dict.items():
                print(f"  Valor original: {value}, Índice: {index}")
