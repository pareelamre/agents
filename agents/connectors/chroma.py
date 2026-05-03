import json
import os
import time

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document

from agents.polymarket.gamma import GammaMarketClient
from agents.utils.objects import SimpleEvent, SimpleMarket


class PolymarketRAG:
    def __init__(self, local_db_directory=None, embedding_function=None) -> None:
        self.gamma_client = GammaMarketClient()
        self.local_db_directory = local_db_directory
        self.embedding_function = embedding_function

    def _documents_from_records(
        self,
        records: "list[dict]",
        content_key: str,
        metadata_keys: "list[str] | None" = None,
    ) -> "list[Document]":
        documents: list[Document] = []
        metadata_keys = metadata_keys or []

        for record in records:
            page_content = str(record.get(content_key, ""))
            metadata = {
                key: record.get(key)
                for key in metadata_keys
                if key in record and record.get(key) is not None
            }
            documents.append(Document(page_content=page_content, metadata=metadata))

        return documents

    def load_json_from_local(
        self, json_file_path=None, vector_db_directory="./local_db"
    ) -> None:
        with open(json_file_path, "r", encoding="utf-8") as input_file:
            records = json.load(input_file)

        loaded_docs = self._documents_from_records(records, content_key="description")

        embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
        Chroma.from_documents(
            loaded_docs, embedding_function, persist_directory=vector_db_directory
        )

    def create_local_markets_rag(self, local_directory="./local_db") -> None:
        all_markets = self.gamma_client.get_all_current_markets()

        if not os.path.isdir(local_directory):
            os.mkdir(local_directory)

        local_file_path = f"{local_directory}/all-current-markets_{time.time()}.json"

        with open(local_file_path, "w+") as output_file:
            json.dump(all_markets, output_file)

        self.load_json_from_local(
            json_file_path=local_file_path, vector_db_directory=local_directory
        )

    def query_local_markets_rag(
        self, local_directory=None, query=None
    ) -> "list[tuple]":
        embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
        local_db = Chroma(
            persist_directory=local_directory, embedding_function=embedding_function
        )
        response_docs = local_db.similarity_search_with_score(query=query)
        return response_docs

    def events(self, events: "list[SimpleEvent]", prompt: str) -> "list[tuple]":
        # create local json file
        local_events_directory: str = "./local_db_events"
        if not os.path.isdir(local_events_directory):
            os.mkdir(local_events_directory)
        local_file_path = f"{local_events_directory}/events.json"
        dict_events = [x.dict() for x in events]
        with open(local_file_path, "w+") as output_file:
            json.dump(dict_events, output_file)

        # create vector db
        loaded_docs = self._documents_from_records(
            dict_events,
            content_key="description",
            metadata_keys=["id", "markets"],
        )
        embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_db_directory = f"{local_events_directory}/chroma"
        local_db = Chroma.from_documents(
            loaded_docs, embedding_function, persist_directory=vector_db_directory
        )

        # query
        return local_db.similarity_search_with_score(query=prompt)

    def markets(self, markets: "list[SimpleMarket]", prompt: str) -> "list[tuple]":
        # create local json file
        local_events_directory: str = "./local_db_markets"
        if not os.path.isdir(local_events_directory):
            os.mkdir(local_events_directory)
        local_file_path = f"{local_events_directory}/markets.json"
        market_records = [
            market.dict() if hasattr(market, "dict") else market for market in markets
        ]
        with open(local_file_path, "w+") as output_file:
            json.dump(market_records, output_file)

        # create vector db
        loaded_docs = self._documents_from_records(
            market_records,
            content_key="description",
            metadata_keys=[
                "id",
                "outcomes",
                "outcome_prices",
                "question",
                "clob_token_ids",
            ],
        )
        embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_db_directory = f"{local_events_directory}/chroma"
        local_db = Chroma.from_documents(
            loaded_docs, embedding_function, persist_directory=vector_db_directory
        )

        # query
        return local_db.similarity_search_with_score(query=prompt)
