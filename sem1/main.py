# main.py

from modules.data_loader import load_data

if __name__ == "__main__":
    file_path = "data/Seminars_1_Group_4.csv"
    data = load_data(file_path)
    
    if data:
        X_train, X_test, y_train, y_test = data
        print("Data loaded and split successfully!")
