import sqlite3
import random
from faker import Faker

# Initialize the Faker library for generating random data
faker = Faker()

def initialize_database():
    connection = sqlite3.connect("flights.db")
    cursor = connection.cursor()

    # Create Table
    table_info = """
    CREATE TABLE IF NOT EXISTS AIRPLANES (
        ID INTEGER PRIMARY KEY AUTOINCREMENT,
        FLIGHT_NUMBER VARCHAR(10),
        AIRLINE VARCHAR(50),
        DESTINATION VARCHAR(50),
        ORIGIN VARCHAR(50),
        DEPARTURE_TIME DATETIME,
        ARRIVAL_TIME DATETIME,
        SEATS_AVAILABLE INT,
        PRICE FLOAT
    );
    """
    cursor.execute(table_info)

    # Generate Sample Data
    sample_data = []
    for _ in range(100):
        flight_number = f"FL{random.randint(1000, 9999)}"
        airline = faker.company()
        origin = faker.city()
        destination = faker.city()
        departure_time = faker.date_time_this_year()
        arrival_time = faker.date_time_this_year()
        seats_available = random.randint(0, 300)
        price = round(random.uniform(50, 500), 2)
        sample_data.append((
            flight_number,
            airline,
            destination,
            origin,
            departure_time,
            arrival_time,
            seats_available,
            price
        ))

    # Insert Sample Data
    cursor.executemany(
        "INSERT INTO AIRPLANES (FLIGHT_NUMBER, AIRLINE, DESTINATION, ORIGIN, DEPARTURE_TIME, ARRIVAL_TIME, SEATS_AVAILABLE, PRICE) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
        sample_data
    )

    print("Database initialized and sample data inserted.")

    # Fetch and display the inserted data
    data = cursor.execute('''SELECT * FROM AIRPLANES''')
    for row in data:
        print(row)

    connection.commit()
    connection.close()

if __name__ == "__main__":
    initialize_database()








