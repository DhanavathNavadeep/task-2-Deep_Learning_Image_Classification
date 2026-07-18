import kagglehub # pyright: ignore[reportMissingImports]

# Download latest version
path = kagglehub.dataset_download("tongpython/cat-and-dog")

print("Path to dataset files:", path)
