def ask_api():
    try:
        # Prompt the user for the API key
        api_key = input("Enter your OCR.Space API key: ")
        return api_key
    except Exception as e:
        print(f"An error occurred during API key input: {str(e)}")
        return None  # Return None if there was an error

def create_env_file():
    try:
        api_key = ask_api()
        if api_key:
            with open('.env', 'w') as f:
                # Write the API key to the .env file in the correct format
                f.write(f"OCR_SPACE={api_key}\n")
            print(".env file created successfully.")
        else:
            print("No API key provided. .env file was not created.")
    except Exception as e:
        print(f"An error occurred during environment file creation: {str(e)}")

def main():
    create_env_file()

if __name__ == "__main__":
    main()
