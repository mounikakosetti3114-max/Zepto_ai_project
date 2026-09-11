import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pandas as pd
import sqlite3

BASE_URL = "https://books.toscrape.com/"

response = requests.get(BASE_URL)
response.encoding = "utf-8"

print("Status:", response.status_code)

# HTML read
soup = BeautifulSoup(response.text, "html.parser")

# Get all categories
categories = []

for link in soup.select(".side_categories ul li ul li a"):
    category_name = link.get_text(strip=True)
    category_url = urljoin(BASE_URL, link.get("href"))

    categories.append({
        "name": category_name,
        "url": category_url
    })

# Select first 5 categories
selected_categories = categories[:5]

print("\nSelected categories:")
for category in selected_categories:
    print(category["name"], category["url"])


# Scrape books from each category
all_books = []

for category in selected_categories:

    print("\nScraping:", category["name"])

    response = requests.get(category["url"])
    response.encoding = "utf-8"
    soup = BeautifulSoup(response.text, "html.parser")

    for book in soup.select("article.product_pod"):

        title = book.h3.a.get("title")
        price = book.select_one(".price_color").get_text(strip=True)
        rating = book.select_one("p.star-rating")["class"][1]
        availability = book.select_one(".availability").get_text(" ", strip=True)

        all_books.append({
            "title": title,
            "price": price,
            "star_rating": rating,
            "availability": availability,
            "category": category["name"]
        })


print("\nTotal books:", len(all_books))

df = pd.DataFrame(all_books)

print(df.head())
print(df.shape)

# Clean price

df["price_gbp"] = (
    df["price"]
    .str.replace("£", "", regex=False)
    .astype(float)
)

df["price_gbp"] = pd.to_numeric(
    df["price_gbp"],
    errors="coerce"
)

# Convert rating words to numbers
rating_map = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}

df["rating"] = df["star_rating"].map(rating_map)


# Convert availability to True/False
df["in_stock"] = df["availability"].str.contains(
    "In stock",
    case=False,
    na=False
)


print(df[[
    "title",
    "price_gbp",
    "rating",
    "in_stock",
    "category"
]].head())

# print("\nMissing values:")
# print(df.isnull().sum())

# print("\nData types:")
# print(df.dtypes)

# print("\nFinal shape:")
# print(df.shape)

df["price_inr"] = df["price_gbp"] * 105.50

print(df[["title", "price_gbp", "price_inr"]].head())
# ===========================================================
# STEP 4: Store into a normalized SQLite database
# ===========================================================

conn = sqlite3.connect("zepto_books.db")
cur = conn.cursor()

cur.executescript("""
DROP TABLE IF EXISTS books;
DROP TABLE IF EXISTS categories;

CREATE TABLE categories (
    category_id   INTEGER PRIMARY KEY,
    category_name TEXT UNIQUE
);

CREATE TABLE books (
    book_id     INTEGER PRIMARY KEY,
    title       TEXT,
    price_gbp   REAL,
    price_inr   REAL,
    rating      INTEGER,
    in_stock    INTEGER,
    category_id INTEGER REFERENCES categories(category_id)
);
""")

# rating లో NaN ఉంటే fill చేయడం (int convert error రాకుండా)
df["rating"] = df["rating"].fillna(df["rating"].median())

category_name_to_id = {}

for category_name in df["category"].unique():
    cur.execute(
        "INSERT INTO categories (category_name) VALUES (?)",
        (category_name,)
    )
    category_name_to_id[category_name] = cur.lastrowid

for _, row in df.iterrows():
    cur.execute("""
        INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        row["title"],
        row["price_gbp"],
        row["price_inr"],
        int(row["rating"]),
        int(row["in_stock"]),
        category_name_to_id[row["category"]]
    ))

conn.commit()

print("\nDatabase created: zepto_books.db")
print("Books inserted:", len(df))

# STEP 5: Run at least 5 SQL queries (incl. one JOIN)

query_1 = """
SELECT title, price_gbp, rating
FROM books
WHERE in_stock = 1 AND price_gbp > 30
ORDER BY price_gbp DESC
LIMIT 10;
"""

query_2 = """
SELECT DISTINCT category_name
FROM categories
ORDER BY category_name;
"""

query_3 = """
SELECT title, price_gbp
FROM books
WHERE price_gbp BETWEEN 20 AND 40
ORDER BY price_gbp ASC;
"""

query_4_join = """
SELECT b.title, b.rating, b.price_gbp, c.category_name
FROM books b
JOIN categories c ON b.category_id = c.category_id
WHERE b.rating IN (4, 5)
ORDER BY c.category_name, b.rating DESC
LIMIT 15;
"""

query_5 = """
SELECT c.category_name, COUNT(*) AS num_books, AVG(b.price_gbp) AS avg_price
FROM books b
JOIN categories c ON b.category_id = c.category_id
GROUP BY c.category_name
ORDER BY avg_price DESC;
"""

result_1 = pd.read_sql(query_1, conn)
result_2 = pd.read_sql(query_2, conn)
result_3 = pd.read_sql(query_3, conn)
result_4_join = pd.read_sql(query_4_join, conn)
result_5 = pd.read_sql(query_5, conn)

print("\n--- Query 1: in-stock, price > 30 ---")
print(result_1)

print("\n--- Query 2: distinct categories ---")
print(result_2)

print("\n--- Query 3: price between 20 and 40 ---")
print(result_3)

print("\n--- Query 4: JOIN, rating 4 or 5 ---")
print(result_4_join)

print("\n--- Query 5: avg price per category (JOIN + GROUP BY) ---")
print(result_5)

# STEP 6: Verify pd.read_sql (SQL JOIN) vs pd.merge (pandas only)

books_table = pd.read_sql("SELECT * FROM books;", conn)
categories_table = pd.read_sql("SELECT * FROM categories;", conn)

merged_df = books_table.merge(categories_table, on="category_id", how="inner")
merged_df = merged_df[merged_df["rating"].isin([4, 5])]
merged_df = merged_df.sort_values(
    ["category_name", "rating"], ascending=[True, False]
).head(15)[["title", "rating", "price_gbp", "category_name"]]

sql_result_sorted = result_4_join.reset_index(drop=True)
merge_result_sorted = merged_df.reset_index(drop=True)

print("\n--- SQL JOIN result (pd.read_sql) ---")
print(sql_result_sorted)

print("\n--- pandas merge result (pd.merge) ---")
print(merge_result_sorted)

print("\nDo both match exactly?", sql_result_sorted.equals(merge_result_sorted))


conn.close()
