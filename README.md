# SQL Query Generation Using Gemini AI

## Overview
This project simplifies database interaction by leveraging Gemini Pro, an advanced AI model, to convert natural language queries into SQL. The application supports both SQLite and MySQL databases and provides a user-friendly interface built with Streamlit.

![image](https://github.com/user-attachments/assets/03137f65-d597-41be-a884-51e130f0e12a)

## Key Features
- Converts natural language questions into SQL queries.
- Supports SQLite and MySQL databases.
- Dynamically retrieves and integrates database schema.
- Provides an intuitive web-based interface.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sql-query-gen.git
   cd sql-query-gen
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   pip install -r requirements.txt
   ```
3. Set up the Gemini API key in a `.env` file:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```
4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Launch the app in your browser.
2. Select the database type (SQLite or MySQL) and provide connection details.
3. Enter a natural language question (e.g., "List all flights from New York to London").
4. View the SQL query and its results displayed in the interface.

 




