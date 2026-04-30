from pathlib import Path

def load_data(path : str | None = None):
    if path is None:

        for label, folder, ext in [('Post-GDPR', '../data/post_gdpr', '*.xml')]:
            post_gdpr_files = list(Path(folder).rglob(ext))
            print(f'{label}: {len(post_gdpr_files)} {ext} files found')
        for label, folder, ext in [('Pre-GDPR', '../data/pre_gdpr', '*.md')]:
            pre_gdpr_files = list(Path(folder).rglob(ext))
            print(f'{label}: {len(pre_gdpr_files)} {ext} files found')

        files = post_gdpr_files + pre_gdpr_files
    else:
        files = list(Path(path).rglob('*.xml')) + list(Path(path).rglob('*.md'))
    return files




if __name__ == "__main__":
    files = load_data()
    # print(files)
    # txt_files = [preprocess_file(str(file)) for file in files]




