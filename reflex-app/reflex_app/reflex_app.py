

# app = rx.App()
# app.add_page(index)

import reflex as rx
from typing import List, Dict
import requests
import os
import urllib.parse


# Set API Key and Base URL
api_key = "iTFqVKUGvfwFc8ZKRaYjNycf"
search_url = "https://www.searchapi.io/api/v1/search"

# Define the state to hold form data and other UI state
class SearchState(rx.State):
    form_data:Dict[str,str] = {
        'age': '',
        'gender': '',
        'style': ''
    }
    merch: List[Dict[str,str]] = [{}]
    loading:bool = False
    error:bool = False
    params = {
            "engine": "google_shopping",
            "q": f"{form_data['style']} clothes for {form_data['gender']} age {form_data['age']}",
            "gl": "us",
            "hl": "en",
            "location": "California,United States",
            "api_key": "iTFqVKUGvfwFc8ZKRaYjNycf",
    }

    def handle_change(self, name, value):
        self.form_data[name] = value

    async def fetch_shirt_data(self, params):
        try:
            response = requests.get(f"{search_url}?{urllib.parse.urlencode(params)}")
            response.raise_for_status()
            data = response.json()
            print("testing ", data)

            self.merch = data['shopping_results']
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
    return rx.vstack(
        rx.heading("Search with Form Data"),
        rx.form(
            rx.input(
                value=SearchState.form_data['age'],
                on_change=lambda value: SearchState.handle_change("age", value),
                placeholder="Age",
                label="Age",
            ),
            rx.input(
                value=SearchState.form_data['gender'],
                on_change=lambda value: SearchState.handle_change("gender", value),
                placeholder="Gender",
                label="Gender",
            ),
            rx.input(
                value=SearchState.form_data['style'],
                on_change=lambda value: SearchState.handle_change("style", value),
                placeholder="Style",
                label="Style",
            ),
            rx.button(
                "Submit",
                on_click=SearchState.submit_form,
                bg="blue",
                color="white"
            ),
            spacing="20px",
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
#return rx.container(
#         rx.color_mode.button(position="top-right"),
#         rx.vstack(
#             rx.heading("Fashion 👕👖", size="9"),

            
#             rx.flex(
#                 rx.foreach(
#                 rx.Var.range(2),
#                 lambda i: rx.card(f"Card {i + 1}", width="16%"),
#             ),
#                 spacing="2",
#                 flex_wrap="wrap",
#                 width="100%",
#             ),
          
#             spacing="5",
#             justify="center",
#             min_height="85vh",
#         ),
#     )