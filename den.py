import pandas as pd

def print_column_names(file_path):
    df = pd.read_excel(file_path)
    print("Mevcut sütun adları:")
    for col in df.columns:
        print(f"- {col}")

def main():
    file_path = 'D:\\data.xlsx'
    print_column_names(file_path)

if __name__ == "__main__":
    main()