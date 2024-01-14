from langchain.llms import GooglePalm
from dotenv import dotenv_values
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain

def main():
    config = dotenv_values(".env")
    llm = GooglePalm(google_api_key=config["PALM_API_KEY"], temperature=0.2)

    db_user = config["DB_USER"]
    db_password = config["DB_PASS"]
    db_host = config["DB_HOST"]
    db_name = config["DB_NAME"]

    db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}",sample_rows_in_table_info=3)

    db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
    # qns1 = db_chain("How many t-shirts do we have left for nike in extra small size and white color?")
    qns3 = db_chain.run("If we have to sell all the Levi’s T-shirts today with discounts applied. How much revenue our store will generate (post discounts)?")

    
if __name__ == "__main__":
    main()