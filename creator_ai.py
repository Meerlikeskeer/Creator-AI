import time
import streamlit as st
import base64
import re
import random
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin, urlparse
import re 

# Initialize OpenAI client
gptClient = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
perplexityClient = OpenAI(api_key=st.secrets["PERPLEXITY_API_KEY"], base_url="https://api.perplexity.ai")

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

keyword = "keyword"
image_link_type = "https://content.instructables.com"
perplexity_instructables_link_type = "https://www.instructables.com/"

introductions = [
        "Hi, I'm your DIY guide, ready to simplify your projects with creative solutions. Feel free to ask me any questions along the way.",
        "Hello, I specialize in making DIY tasks easy, even when you're missing materials. I’m here to answer your questions anytime.",
        "Hey there, I’m here to guide you step-by-step to finish any project quickly. Ask me anything if you need help.",
        "Hi, I’m your go-to creator for solving DIY challenges with minimal resources. Feel free to reach out with any questions.",
        "Hello, I focus on efficient, no-nonsense solutions for your DIY needs. Don’t hesitate to ask if you’re unsure about something.",
        "Hey, I simplify projects and teach you how to work with what you have on hand. Questions are always welcome!",
        "Hi, I can help you complete projects creatively, even with limited tools or supplies. Let me know if you need clarity or guidance.",
        "Hello, I’m here to ensure you can finish your DIY tasks with ease and clarity. I’m happy to answer any of your questions.",
        "Hey, I’ll guide you to quick, effective results in your DIY endeavors. If you’re stuck, just ask!",
        "Hi, I specialize in teaching you how to make things work, no matter the constraints. Your questions are always encouraged."
        ]
# Streamlit UI
st.title("Creator.ai")


def is_valid_instructables_url(url: str, base_url: str = "https://www.instructables.com/") -> bool:
    """
    Check if the given URL matches the Instructables format.
    
    Args:
        url (str): URL to validate
        base_url (str): Base URL to check against
        
    Returns:
        bool: True if URL matches format, False otherwise
    """
    # Check if URL starts with the base URL
    if not url.startswith(base_url):
        return False
    
    # Check if URL has content after the base URL
    if len(url) <= len(base_url):
        return False
    
    # Check if URL contains only valid characters after base URL
    path = url[len(base_url):]
    valid_path_pattern = r'^[a-zA-Z0-9-]+/$'
    
    return bool(re.match(valid_path_pattern, path))

@st.cache_data
def scrape_first_instructables_url(text: str, base_url: str = "https://www.instructables.com/") -> str:
    """
    Extract the first valid Instructables URL from anywhere in the given text.
    
    Args:
        text (str): Text to search for URLs
        base_url (str): Base URL to validate against
        
    Returns:
        str: First valid Instructables URL found, or empty string if none found
    """
    # Find all URLs starting with https:// or http://
    pattern = r'https?://[^\s<>"\']+'
    all_urls = re.findall(pattern, text)
    
    # Filter for Instructables URLs and validate them
    for url in all_urls:
        # Remove any trailing punctuation or characters that might have been caught
        url = re.sub(r'[.,!?)]$', '', url)
        
        # Check if it's a valid Instructables URL
        if url.startswith(base_url) and is_valid_instructables_url(url, base_url):
            return url
    
    return ""


def perplexity_search_steps(userInput, instructablesLink):
    messages = [
            {
                "role": "system",
                "content": (
                    "You are a master DIY creator and can search for the project the user asks. You love to teach and explain clearly."
            ),
                },
                {   
                "role": "user",
                "content": (
                    f"Give me steps on how to build {userInput}. Get this information from Instructables at this link {instructablesLink}. If there is no link, then search the web and make an attempt to generate steps. Make sure you give steps for one project, not multiple types of the same project."
                ), 
            },
        ]

    # chat completion without streaming
    response = perplexityClient.chat.completions.create(
        model="llama-3.1-sonar-large-128k-online",
        messages=messages,
    )
    print(response.choices[0].message.content)

    return response.choices[0].message.content

def extract_materials(search_result):
    response = gptClient.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a master DIY creator. You love to teach and explain clearly."
                },
                {
                    "role": "user",
                    "content": [
            {
                "type": "text",
                "text": f"These are the instructions for a project {search_result}. Extract and return only the materials needed in bullet point format."
        }
    ]
                }
            ],
            temperature=0.5,
            max_tokens=2048,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
    print(response.choices[0].message.content)
    return response.choices[0].message.content
@st.cache_data
def user_question_response(steps, prompt, image_decoded, image_paths):
        if image_decoded is None:
            response1 = gptClient.chat.completions.create(
                model="gpt-4o-mini",
                messages=[ {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a master DIY creator. You love to teach and understand how to help if materials are missing. You can see images and search for images to respond. Just give the simplest way to finish the projects the user asks. Do not add additional instructions for anything which is optional, or not asked for.  Upload any useful image from the web which is relevant to the users questions."
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Here are the steps for the project: {steps}. Here is the user's question: {prompt} The user has not provided any image, here is our database of image urls {image_paths}. Search the URLs to try to answer the User's Question and feel free to display any specific image from the urls or from your searching.",
                        }
                    ],
                }],
                temperature=1.25,
                max_tokens=2048,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
        else:
            response1 = perplexityClient.chat.completions.create(
                model="gpt-4o-mini",
                messages=[ {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a master DIY creator. You love to teach and understand how to help if materials are missing. You can see images, analyze and interpret them. Just give the simplest way to finish the projects the user asks. Do not add additional instructions for anything which is optional, or not asked for. Upload any useful image from the web which is relevant to the users questions."
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Here are the steps for the project: {steps}. Here is the user's question: {prompt} The first image is what the user has built, here is our database of image urls {image_paths}. Search the URLs to try to answer the User's Question and feel free to display a specific image from the from the urls or from your searching.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url":  f"data:image/jpeg;base64,{image_decoded}"
                            },
                        }
                    ],
                }],
                temperature=0.75,
                max_tokens=2048,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
        print(response1.choices[0].message.content)
        
        return response1.choices[0].message.content

def filter_instructables_links(links):
    """
    Filter a list of strings to keep only those containing 'https://content.instructables.com'
   
    Args:
        links (list): List of strings to filter
       
    Returns:
        list: Filtered list containing only strings with the instructables content URL
    """
    return [link for link in links if image_link_type in link]

def download_image(url, folder):
    # Get the file name
    filename = os.path.join(folder, url.split("/")[-1])
   
    # Download the image
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {filename}")
    else:
        print(f"Failed to download: {url}")

@st.cache_data
def extract_images(url):
    # Send a GET request to the webpage
    response = requests.get(url)
   
    # Parse the HTML content
    soup = BeautifulSoup(response.content, 'html.parser')
   
    # Find all img tags
    img_tags = soup.find_all('img')
   
    # Extract image URLs
    image_urls = []
    for img in img_tags:
        img_url = img.get('src')
        if img_url:
            # Make sure the URL is absolute
            img_url = urljoin(url, img_url)
            image_urls.append(img_url)
                
    return image_urls

@st.cache_data
def perplexity_search_link(userInput):
    example_url = "https://content.instructables.com/FBQ/H9I5/J8F5YPYE/FBQH9I5J8F5YPYE.jpg?auto=webp&fit=bounds&frame=1&height=1024&width=1024"

    messages = [
        {
        "role": "system",
        "content": (
            "You are a master DIY creator and can search for the project the user asks. You love to teach and explain clearly."
        ),
        },
        {   
        "role": "user",
        "content": (
            f"Give me the link to a instructables page on how to build the users requested project in real life. The user inputed this {userInput} If the project is on instructables, output only one of the most relevant and recently created instructables link. Make sure you only ouput the link similar to this {perplexity_instructables_link_type} and nothing else."
            ), 
        },
    ]
    
    # chat completion without streaming
    response = perplexityClient.chat.completions.create(
        model="llama-3.1-sonar-large-128k-online",
        messages=messages,
    )
    print("TESTING PERPLEXITY: "+response.choices[0].message.content)
    return response.choices[0].message.content

def filter_strings_by_majority_length(strings):
    # If the list is empty, return empty list
    if not strings:
        return []
   
    # Get all string lengths
    lengths = [len(s) for s in strings]
   
    # Sort lengths to find the majority length
    # Using the middle element as majority length since we want to keep longer strings
    majority_length = sorted(lengths)[len(lengths)//2]
   
    # Filter strings keeping only those with length >= majority_length
    majority_length = majority_length - 15

    filtered_strings = [s for s in strings if len(s) >= majority_length]
   
    return filtered_strings
@st.cache_data
def display_project_results(is_valid_url, url, search_results, length_filtered_urls):
    """
    Cached function to display project instructions and images
    
    Args:
        is_valid_url (bool): Whether the Instructables URL is valid
        url (str): The project URL
        search_results (str): Extracted project instructions
        length_filtered_urls (list): List of filtered image URLs
    
    Returns:
        tuple: Columns configuration for Streamlit display
    """
    if is_valid_url:
        print("IF STATEMENT")
        steps_column, images_column = st.columns(2, vertical_alignment="top")
        images_column.header("Images")
        steps_column.header("Instructions")
        steps_column.write(search_results) 
        
        for i in length_filtered_urls:
            # Display each image in the sidebar
            images_column.image(i, width=500)
        
        # return steps_column, images_column
    
    else:
        print("ELSE STATEMENT")
        steps_column = st.columns(1, vertical_alignment="top")
        steps_column.header("Instructions")
        steps_column.write(search_results)
        
        # return steps_column, None
    
@st.cache_data
def process_diy_project(userInput, perplexity_instructables_link_type):
    """
    Cached function to process DIY project search and extract information
    
    Args:
        userInput (str): User's project description
        perplexity_instructables_link_type (str): Type of Instructables link
    
    Returns:
        tuple: Processed search results and image URLs
    """
    url = scrape_first_instructables_url(
        perplexity_search_link(userInput), 
        perplexity_instructables_link_type
    )
    
    search_results = perplexity_search_steps(userInput, url)
    
    length_filtered_urls = []
    if is_valid_instructables_url(url):
        scraped_unfiltered_urls = extract_images(url)
        scraped_filtered_urls = filter_instructables_links(scraped_unfiltered_urls)
        length_filtered_urls = filter_strings_by_majority_length(scraped_filtered_urls)
    
    return url, search_results, length_filtered_urls
def clear_text():
    st.session_state["chat"] = ""
def chatbot_respond(search_results, length_filtered_urls): 
    # Initialize session state for messages and images
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    # Container for chat messages (history remains visible)
    chat_container = st.container()
    chat_container.chat_message("assistant").write(f"Creator: {random.choice(introductions)}")
    # **Fixed input section at the bottom**
    bottom_container = st.empty()
    with bottom_container.container():
        uploaded_image = st.file_uploader(
            "Upload an image of your creation", 
            type=['jpg', 'jpeg', 'png'], 
            help="Upload an image if you need help with your Creation"
        )
        chat = st.text_input("Ask about your creation here", key="text")
    # Process user input when the button is clicked
    if chat:
        # Process uploaded image if available
        if uploaded_image is not None:
            base64_uploaded_image = base64.b64encode(uploaded_image.getvalue()).decode('utf-8')
            response = user_question_response(search_results, chat, base64_uploaded_image, length_filtered_urls)
        else:
            response = user_question_response(search_results, chat, None, length_filtered_urls)

        # Append messages to session state
        st.session_state.messages.append({"role": "user", "content": chat})
        st.session_state.messages.append({"role": "assistant", "content": f"Creator: {response}"})
    with chat_container:
        for msg in st.session_state.messages:
            st.chat_message(msg['role']).write(msg['content'])
    clear_text()

# Main app
def main():

    userInput = st.text_area("Describe your DIY project")
    if st.text_input and userInput != "": 
        st.header(userInput)

        url, search_results, length_filtered_urls = process_diy_project(
        userInput, 
        perplexity_instructables_link_type
        )

        st.markdown("""
        <style>
        .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 75%;
        }
        </style>
        """, unsafe_allow_html=True)

        display_project_results(
            is_valid_instructables_url(url), 
            url, 
            search_results, 
            length_filtered_urls
        )

        st.header("Need Help?")
        chatbot_respond(search_results, length_filtered_urls)
        
main()
