from src.data_loader import load_data
from src.embedding_model import get_embeddings
from src.recommender import SustainableRecommender

DATA_PATH = "data/processed/cleaned_flipkart_data.csv   "

def main():
    df = load_data(DATA_PATH)
    embeddings = get_embeddings(df)

    recommender = SustainableRecommender(df, embeddings)

    # 👇 choose ANY valid product index
    product_index = 0
    print("INPUT PRODUCT:")
    print(df.loc[product_index, [
        "product_name",
        "brand",
        "product_category_tree",
        "retail_price"
    ]])

    results = recommender.recommend_alternatives(
        product_index=product_index,
        top_k=5
    )

    print("\nRecommended Alternatives:\n")
    print(results)

if __name__ == "__main__":
    main()
