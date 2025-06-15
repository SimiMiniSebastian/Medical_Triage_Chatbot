from chatbot_utils import load_data, train_model

def main():
    df = load_data()
    model, label_encoder = train_model(df)
    print("Model trained and saved successfully.")

if __name__ == "__main__":
    main()
