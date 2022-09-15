import dvc.api


if __name__ == '__main__':
    params = dvc.api.params_show()
    for key, value in params.items():
        print(f"{key}: {value}")