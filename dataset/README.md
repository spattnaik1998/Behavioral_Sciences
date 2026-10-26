# Raw data

The raw review files come from Datafiniti's "Consumer Reviews of Amazon Products" dataset on Kaggle (https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products). Download them and place them here:

- `Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv` (main data; 28,332 reviews)
- `1429_1.csv` (external set: Kindle, Fire and Echo reviews)
- `Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv` (the 5,000-row file the first version described by mistake)

You do not need them to reproduce the paper: `data/` already holds the derived files (`may19_slim.csv.gz`, `pool.csv`, `external_sample.csv`, and the annotation sets). Check the dataset's license on Kaggle before redistributing it.
