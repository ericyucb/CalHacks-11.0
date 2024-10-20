

# app = rx.App()
# app.add_page(index)

import reflex as rx
from typing import List, Dict
import requests
import os
import urllib.parse
import aiohttp
import asyncio
import cv2
from PIL import Image
from io import BytesIO
from shirt_fitter.mastercamera import generateSegmentation, master

from shirt_fitter.gemini import aura



search_url = "https://www.searchapi.io/api/v1/search"



# Define the state to hold form data and other UI state
class SearchState(rx.State):
    merch: List[Dict[str,str]] = [{}]
    loading:bool = False
    error:bool = False
    recommendations: str = []
    description: str = ""




    params = {
            "engine": "google_shopping",
            "q": "Uniqlo Red Airism Tee",
            "gl": "us",
            "location": "California,United States",
            "num": "1",
            "api_key": "hVgDDfo4YQNdBaet1KDpNq8G",
    }



    def try_on(self):
        master("shirt_fitter/test_clothes/shirt0.png","shirt_fitter/test_clothes/shirt1.png","shirt_fitter/test_clothes/shirt2.png")

    def create_aura(self):
        
        output = aura()
        if output:
            description, recommendation = output
            self.description = description
            for i in range(3):
                shirt_data = self.fetch_shirt_data(recommendation[i])
                print("testing shirt_data returns ", shirt_data["shopping_results"])
                if shirt_data:
                    individual_shirt = shirt_data["shopping_results"][0]
                    url = individual_shirt["thumbnail"]
                    img = requests.get(url)
                    if img.status_code == 200:
                        # Open a file in write-binary mode and save the image
                        img = Image.open(BytesIO(img.content))
                        img.save("shirt_fitter/test_clothes/shirt" + str(i)+".png")
                        generateSegmentation("shirt_fitter/test_clothes/shirt" + str(i)+".png")
                        print(f"Image saved as shirt"+str(i))
                        
                    else:
                        print(f"Failed to retrieve image. Status code: {img.status_code}")
                    self.merch.append(individual_shirt)  
        else:
            self.create_aura()



    def fetch_shirt_data(self, shirt_name):
        try:
            self.params["q"] = shirt_name + " white background, shirt only"
            response = requests.get(f"{search_url}?{urllib.parse.urlencode(self.params)}")
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as err:
            print(err)
            self.error = True
        finally:
            self.loading = False

def create_product_image(alt_text, image_src):
    """Create a product image with specific dimensions and styling."""
    return rx.image(
        src=image_src,
        alt=alt_text,
        height="16rem",
        object_fit="cover",
        width="100%",
    )
def create_heading(text):
    """Create a styled heading with specific font properties."""
    return rx.heading(
        text,
        font_weight="600",
        color="#1F2937",
        font_size="1.25rem",
        line_height="1.75rem",
        as_="h3",
    )
def create_description_text(text):
    """Create a styled description text for products."""
    return rx.text(
        text, margin_top="0.5rem", color="#4B5563"
    )

def create_price_and_cart_container(price):
    """Create a container with price and 'Add to Cart' button."""
    return rx.flex(
        create_price_text(price=price),
        display="flex",
        align_items="center",
        justify_content="space-between",
        margin_top="1rem",
    )



def create_price_text(price):
    """Create a styled price text for products."""
    return rx.text.span(
        "$"+price,
        font_weight="700",
        color="#1F2937",
        font_size="1.25rem",
        line_height="1.75rem",
    )

def create_product_details_box(title, description, price):
    """Create a box containing product details including title, description, and price."""
    return rx.box(
        create_heading(text=title),
        create_description_text(text=description),
        create_price_and_cart_container(price=price),
        padding="1rem",
    )

def get_product(product):
    return rx.box(
            create_product_image(
                alt_text="No Shirts Yet",
                image_src=product['thumbnail'],
            ),
            create_product_details_box(
                title=product['title'],
                description="",
                price=product['extracted_price'],
            ),
          
            rx.el.button(
                "Add to Cart",
                
                background_color="#3B82F6",
                _hover={"background-color": "#2563EB"},
                padding_left="1rem",
                padding_right="1rem",
                padding_top="0.5rem",
                padding_bottom="0.5rem",
                border_radius="0.25rem",
                color="#ffffff", on_click=rx.redirect(product["product_link"])
        ),
          background_color="#ffffff",
            overflow="hidden",
            border_radius="0.5rem",
            box_shadow="0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)",
            )
      
    
    # return rx.vstack(
    #     rx.heading(
    #         product['title']
    #     ),
    #     rx.image(src=product['thumbnail'], alt=product['title'], width=350, height=350),
    #     rx.link("Buy ", product['product_link'])

    # )
    
def create_upload_section():
    """Create the main upload section with heading, description, and form."""
    return rx.box(
        rx.heading(
            "Get Personalized Fashion Recommendations",
            font_weight="700",
            margin_bottom="1.5rem",
            font_size="1.875rem",
            line_height="2.25rem",
            color="#1E40AF",
            text_align="center",
            as_="h2",
        ),
        rx.text(
            "Upload a photo of yourself and let our AI stylist provide customized fashion advice and outfit suggestions.",
            margin_bottom="2rem",
            color="#2563EB",
            text_align="center",
        ),
        margin_bottom="3rem",
    )

def create_section_heading():
    """Create a section heading with specific styling."""
    return rx.heading(
        "Shirt AI",
        font_weight="600",
        margin_bottom="1rem",
        color="#1E40AF",
        font_size="1.25rem",
        line_height="1.75rem",
        as_="h3",
    )
# Define the UI layout
def index():
  


    return rx.container(
        rx.color_mode.button(position="top-right"),

        rx.vstack(
            create_section_heading(),
            create_upload_section(),
        ),
        rx.button(
            "Take Photo 📸",
            on_click=SearchState.create_aura
        ),
        
            rx.button(
                "Try on clothes 👕",
                on_click=SearchState.try_on
            ),
        rx.cond(SearchState.loading, rx.text("Loading...")),
        rx.cond(SearchState.error, rx.text(f"Error: {SearchState.error}")),
        rx.cond(SearchState.description,
                rx.text(SearchState.description)),
        rx.cond(
            SearchState.merch,
            rx.vstack(
                rx.heading("Search Results:"),
                rx.box(
                    rx.foreach(SearchState.merch, get_product)

                )
                
            )
        ),
    )



#Define the Reflex app
app = rx.App()
app.add_page(index)

