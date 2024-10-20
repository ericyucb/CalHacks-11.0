

# app = rx.App()
# app.add_page(index)

import reflex as rx
from typing import List, Dict
import requests
import os
import urllib.parse

from shirt_fitter.gemini import aura


# Set API Key and Base URL
api_key = "iTFqVKUGvfwFc8ZKRaYjNycf"
search_url = "https://www.searchapi.io/api/v1/search"

# Define the state to hold form data and other UI state
class SearchState(rx.State):

    merch: List[Dict[str,str]] = [{1: ["Test"], 2: ["Test"]}]
    loading:bool = False
    error:bool = False
    recommendations: str = []
    description: str = ""

    params = {
            "engine": "google_shopping",
            "q": "Uniqlo Red Airism Tee",
            "gl": "us",
            "hl": "en",
            "location": "California,United States",
            "api_key": "iTFqVKUGvfwFc8ZKRaYjNycf",
    }

    # def handle_change(self, name, value):
    #     self.form_data[name] = value

    def create_aura(self):
        output = aura()
        if output:
            description, recommendations = output
            [fetch_shirt_data(params, )for item in recommendations]
        else:
            self.create_aura()


    async def fetch_shirt_data(self, params, shirt_name):
        try:
            params.q = shirt_name
            response = requests.get(f"{search_url}?{urllib.parse.urlencode(params)}")
            response.raise_for_status()
            data = response.json()
            print("testing ", data)

            self.merch = data['shopping_results'][0]
            print("testing ", self.merch)
        except requests.exceptions.RequestException as err:
            print(err)
            self.error = True
        finally:
            self.loading = False

    async def submit_form(self):
        self.loading = True
        self.error = False
        await self.fetch_shirt_data(self.params)

def get_product(product):
    return rx.image(src=product['thumbnail'], alt=product['title'], width=350, height=350)
    
    

# Define the UI layout
def index():
    return  rx.container(
        rx.color_mode.button(position="top-right"),
        rx.vstack(
            rx.heading("Fashion 👕👖", size="30"),
            rx.flex(
                rx.foreach(
                rx.Var.range(2),
                lambda i: rx.card(f"Card {i + 1}", width="16%"),
            ),
                spacing="2",
                flex_wrap="wrap",
                width="100%",
            ),
          
            spacing="5",
            justify="center",
            min_height="85vh",
        ),
        rx.button(
            "Take Photo",
            on_click=SearchState.create_aura
        ),
        rx.cond(SearchState.loading, rx.text("Loading...")),
        rx.cond(SearchState.error, rx.text(f"Error: {SearchState.error}")),
        rx.cond(
            SearchState.merch,
            rx.vstack(
                rx.heading("Search Results:"),
                
                rx.foreach(SearchState.merch, get_product)
                
            )
        ),
    )

# Define the Reflex app
app = rx.App()
app.add_page(index)
# app.compile()
