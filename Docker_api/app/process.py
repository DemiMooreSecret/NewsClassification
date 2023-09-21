import pandas as pd 

def printApp(app, msg):
    app.status += msg + "\n"

def categories(file_path, app_main):
    df = pd.read_csv(file_path)

    # Добавьте фильтрацию и категории
    
    printApp(app_main, "Completed!")
    return df