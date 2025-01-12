import streamlit as st
import sqlite3
import mysql.connector
import pandas as pd
from dotenv import load_dotenv
import os
import google.generativeai as genai

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Define the Gemini model
model = genai.GenerativeModel('gemini-pro')

# Helper function to list tables in SQLite database
def list_tables_sqlite(db):
    try:
        conn = sqlite3.connect(db)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cur.fetchall()
        conn.close()
        return [table[0] for table in tables]
    except Exception as e:
        raise Exception(f"Error retrieving tables: {e}")

# Helper function to list tables in MySQL database
def list_tables_mysql(host, user, password):
    try:
        conn = mysql.connector.connect(
            host=host, user=user, password=password
        )
        cur = conn.cursor()
        cur.execute("SHOW DATABASES;")
        databases = cur.fetchall()
        conn.close()
        return [db[0] for db in databases]
    except Exception as e:
        raise Exception(f"Error retrieving databases: {e}")

# Helper function to get table schema for SQLite
def get_table_schema_sqlite(db, table_name):
    try:
        conn = sqlite3.connect(db)
        cur = conn.cursor()
        cur.execute(f"PRAGMA table_info({table_name});")
        schema = cur.fetchall()
        conn.close()
        return schema
    except Exception as e:
        raise Exception(f"Error retrieving schema for table {table_name}: {e}")

# Helper function to get table schema for MySQL
def get_table_schema_mysql(host, user, password, database, table_name):
    try:
        conn = mysql.connector.connect(
            host=host, user=user, password=password, database=database
        )
        cur = conn.cursor()
        cur.execute(f"DESCRIBE {table_name};")
        schema = cur.fetchall()
        conn.close()
        return schema
    except Exception as e:
        raise Exception(f"Error retrieving schema for table {table_name}: {e}")

# Function to dynamically build schema prompt
def build_schema_prompt(db_type, **kwargs):
    try:
        if db_type == "SQLite":
            tables = list_tables_sqlite(kwargs['db'])
            if not tables:
                return "The SQLite database is empty."

            schema_prompt = "The database schema is as follows:\n"
            for table in tables:
                schema_prompt += f"- Table: {table} ("
                schema = get_table_schema_sqlite(kwargs['db'], table)
                columns = [f"{col[1]} {col[2]}" for col in schema]  # column name and type
                schema_prompt += ", ".join(columns) + ")\n"
            return schema_prompt

        elif db_type == "MySQL":
            tables = kwargs.get('tables', [])
            if not tables:
                return "The MySQL database is empty."

            schema_prompt = "The database schema is as follows:\n"
            for table in tables:
                schema_prompt += f"- Table: {table[0]} ("
                schema = get_table_schema_mysql(kwargs['host'], kwargs['user'], kwargs['password'], kwargs['database'], table[0])
                columns = [f"{col[0]} {col[1]}" for col in schema]  # column name and type
                schema_prompt += ", ".join(columns) + ")\n"
            return schema_prompt

        else:
            return "Unsupported database type."
    except Exception as e:
        return f"Error building schema prompt: {e}"

# Prompt for generating SQL queries
def get_prompt_with_schema(db_type, **kwargs):
    schema_prompt = build_schema_prompt(db_type, **kwargs)
    base_prompt = """
You are an expert assistant specialized in generating SQL queries from natural language questions. {schema}
Your goal is to create highly accurate SQL queries tailored to this schema. Avoid making assumptions about non-existent tables or columns.
Formatting Rules:
Ensure the query is properly formatted and syntactically correct.
Avoid unnecessary columns or ambiguous references.
Use aliases for clarity when joining tables.
Include LIMIT if the question implies a need to show limited results (e.g., "Show the first 5 records").
"""
    return base_prompt.format(schema=schema_prompt)

# Define Gemini response function
def get_gemini_response(question, db_type, **kwargs):
    try:
        prompt = get_prompt_with_schema(db_type, **kwargs)
        response = model.generate_content(prompt + question)
        return response.text
    except Exception as e:
        raise Exception(f"AI Error: {e}")

# Function to execute and read SQL query for SQLite
def read_sql_query_sqlite(sql, db):
    try:
        conn = sqlite3.connect(db)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        # Fetch column names from the cursor
        columns = [description[0] for description in cur.description]
        conn.commit()
        conn.close()
        return rows, columns
    except sqlite3.OperationalError as e:
        if "no such table" in str(e):
            raise Exception(f"Table not found error: {e}. Please verify the database schema.")
        raise Exception(f"SQLite Error: {e} | Query: {sql}")
    except sqlite3.Error as e:
        raise Exception(f"SQLite Error: {e} | Query: {sql}")
    except Exception as e:
        raise Exception(f"General Error: {e} | Query: {sql}")

# Function to execute and read SQL query for MySQL
def read_sql_query_mysql(sql, host, user, password, database):
    try:
        conn = mysql.connector.connect(
            host=host, user=user, password=password, database=database
        )
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        # Fetch column names from the cursor
        columns = [description[0] for description in cur.description]
        conn.commit()
        conn.close()
        return rows, columns
    except mysql.connector.Error as e:
        raise Exception(f"MySQL Error: {e} | Query: {sql}")
    except Exception as e:
        raise Exception(f"General Error: {e} | Query: {sql}")

# Function to clean up SQL query
def clean_sql_query(query):
    # Remove any backticks, markdown syntax, or extra whitespace
    cleaned_query = query.strip().replace("```sql", "").replace("```", "").replace("`", "")
    return cleaned_query.strip()

# Streamlit app setup
st.title("AI-Powered SQL Query Generator and Executor")

# Database selection
db_type = st.selectbox("Select Database Type:", ["SQLite", "MySQL"], index=0)

databases = []
selected_database = None
if db_type == "SQLite":
    db_path = st.text_input("SQLite Database Path:", "flights.db")
    if db_path:
        try:
            tables = list_tables_sqlite(db_path)
            st.markdown("### Tables in SQLite Database:")
            if tables:
                st.write(tables)
            else:
                st.write("No tables found in the SQLite database.")
        except Exception as e:
            st.error(f"Error: {e}")

elif db_type == "MySQL":
    mysql_host = st.text_input("MySQL Host:", "localhost")
    mysql_user = st.text_input("MySQL User:", "root")
    mysql_password = st.text_input("MySQL Password:", type="password")

    if mysql_host and mysql_user and mysql_password:
        try:
            databases = list_tables_mysql(mysql_host, mysql_user, mysql_password)
            selected_database = st.selectbox("Select MySQL Database:", databases)

            if selected_database:
                conn = mysql.connector.connect(
                    host=mysql_host, user=mysql_user, password=mysql_password, database=selected_database
                )
                cur = conn.cursor()
                cur.execute("SHOW TABLES;")
                tables = cur.fetchall()
                st.markdown("### Tables in MySQL Database:")
                if tables:
                    st.write([table[0] for table in tables])
                else:
                    st.write("No tables found in the selected MySQL database.")
                conn.close()
        except Exception as e:
            st.error(f"Error: {e}")

# User input for natural language question
question = st.text_input("Enter your question:", placeholder="e.g., How many airplanes are in the database?", key="input")
submit = st.button("Ask the question")

if submit:
    if not question.strip():
        st.error("Please enter a valid question.")
    else:
        try:
            if db_type == "SQLite":
                # Generate SQL query using AI
                response = get_gemini_response(question, "SQLite", db=db_path)
                cleaned_query = clean_sql_query(response)
                st.markdown("### Generated SQL Query:")
                st.code(cleaned_query, language="sql")

                # Execute SQL query
                try:
                    data, columns = read_sql_query_sqlite(cleaned_query, db_path)
                    if data:
                        df = pd.DataFrame(data, columns=columns)
                        st.markdown("### Query Results:")
                        st.dataframe(df)
                    else:
                        st.success("Query executed successfully but returned no results.")
                except Exception as e:
                    st.error(f"Execution Error: {e}")

            elif db_type == "MySQL" and selected_database:
                # Generate SQL query using AI
                response = get_gemini_response(question, "MySQL", host=mysql_host, user=mysql_user, password=mysql_password, database=selected_database, tables=tables)
                cleaned_query = clean_sql_query(response)
                st.markdown("### Generated SQL Query:")
                st.code(cleaned_query, language="sql")

                # Execute SQL query
                try:
                    data, columns = read_sql_query_mysql(cleaned_query, mysql_host, mysql_user, mysql_password, selected_database)
                    if data:
                        df = pd.DataFrame(data, columns=columns)
                        st.markdown("### Query Results:")
                        st.dataframe(df)
                    else:
                        st.success("Query executed successfully but returned no results.")
                except Exception as e:
                    st.error(f"Execution Error: {e}")

        except Exception as e:
            st.error(f"AI Error: {e}")
