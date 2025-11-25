import json
import random
import re
import string
from faker import Faker
from num2words import num2words

# Initialize Faker
fake = Faker()

# STT Noise Mappings
DIGIT_MAP = {
    '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
    '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
}

# Context templates to surround entities
TEMPLATES = {
    "CREDIT_CARD": [
        "my credit card number is {val}",
        "charge it to {val} please",
        "card ending in {val}",
        "here is the number {val} thanks",
        "use the card {val} for payment"
    ],
    "PHONE": [
        "call me at {val}",
        "my number is {val}",
        "reach me on {val} tomorrow",
        "contact number {val}",
        "phone is {val}"
    ],
    "EMAIL": [
        "email me at {val}",
        "send it to {val}",
        "my address is {val}",
        "contact {val} for details",
        "cc {val} on the reply"
    ],
    "PERSON_NAME": [
        "my name is {val}",
        "this is {val} speaking",
        "ask for {val} at the desk",
        "is {val} available",
        "meeting with {val}"
    ],
    "DATE": [
        "born on {val}",
        "schedule it for {val}",
        "date is {val}",
        "deadline is {val}",
        "happened on {val}"
    ],
    "CITY": [
        "i live in {val}",
        "traveling to {val}",
        "weather in {val}",
        "from {val} originally",
        "near {val}"
    ],
    "LOCATION": [
        "meet at {val}",
        "office is at {val}",
        "located in {val}",
        "go to {val}",
        "address is {val}"
    ]
}

def clean_text(text):
    """Removes punctuation and lowercases text to mimic STT."""
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.lower()

def noise_digits(text):
    """Converts '123' -> 'one two three'."""
    # Split by spaces to handle mixed text, though usually applied to pure numbers
    noisy = []
    for char in text:
        if char in DIGIT_MAP:
            noisy.append(DIGIT_MAP[char])
        elif char in [' ', '-']: 
            continue # Skip separators in phone/cards
        else:
            noisy.append(char)
    return " ".join(noisy)

def noise_email(text):
    """Converts 'john@gmail.com' -> 'john at gmail dot com'."""
    text = text.replace("@", " at ")
    text = text.replace(".", " dot ")
    return clean_text(text)

def noise_date(date_obj):
    """Converts date object to various spoken formats."""
    formats = [
        "%B %d %Y", # January 01 2023
        "%d %B %Y", # 01 January 2023
        "%A %B %d"  # Monday January 01
    ]
    # Pick a random format
    fmt = random.choice(formats)
    text = date_obj.strftime(fmt)
    
    # Convert the year digits to words if present (e.g., 2023 -> twenty twenty three)
    # For simplicity here, we just lowercase. 
    # Ideally, you'd use num2words on the year, but standard STT sometimes leaves years as digits 
    # or converts them. Let's just clean it for now or use num2words for full accuracy.
    return clean_text(text)

def generate_entity(label):
    """Generates a raw entity value and its noisy STT version."""
    
    if label == "CREDIT_CARD":
        raw = fake.credit_card_number()
        noisy = noise_digits(raw)
        
    elif label == "PHONE":
        raw = fake.phone_number()
        # Strip extension if present for simplicity
        raw = raw.split('x')[0]
        noisy = noise_digits(raw)
        
    elif label == "EMAIL":
        raw = fake.email()
        noisy = noise_email(raw)
        
    elif label == "PERSON_NAME":
        raw = fake.name()
        noisy = clean_text(raw)
        
    elif label == "DATE":
        raw_date = fake.date_between(start_date='-5y', end_date='+5y')
        noisy = noise_date(raw_date)
        
    elif label == "CITY":
        raw = fake.city()
        noisy = clean_text(raw)
        
    elif label == "LOCATION":
        # Address often contains numbers, we should noise those too
        raw = fake.street_address()
        # Simple approach: just lower and remove punctuation, keep numbers as digits 
        # OR convert numbers. STT is inconsistent. Let's convert digits for consistency.
        # But distinct from phone/card, address numbers are read as full numbers often.
        # Let's just clean punctuation for Loc/City to save complexity.
        noisy = clean_text(raw)
        
    else:
        return None, None

    return label, noisy

def generate_sample(idx, output_file):
    """Creates a single training example."""
    
    # 80% chance to include an entity, 20% pure noise/filler (Negative examples)
    if random.random() < 0.8:
        label = random.choice(list(TEMPLATES.keys()))
        _, entity_text = generate_entity(label)
        
        template = random.choice(TEMPLATES[label])
        
        # Split template into before/after
        parts = template.split("{val}")
        prefix = clean_text(parts[0])
        suffix = clean_text(parts[1])
        
        # Construct full text
        # Note: Extra spaces handled by join, then stripping
        full_text = f"{prefix} {entity_text} {suffix}".strip()
        # Fix double spaces
        full_text = re.sub(r'\s+', ' ', full_text)
        
        # Find start/end
        # We search for the entity text in the full text to get exact indices
        start_idx = full_text.find(entity_text)
        end_idx = start_idx + len(entity_text)
        
        entities = [{
            "start": start_idx,
            "end": end_idx,
            "label": label
        }]
    else:
        # Negative sample (No entities)
        full_text = clean_text(fake.sentence(nb_words=10))
        entities = []

    sample = {
        "id": f"gen_{idx}",
        "text": full_text,
        "entities": entities
    }
    
    return sample

def create_dataset(filename, count):
    print(f"Generating {count} examples for {filename}...")
    with open(filename, 'w', encoding='utf-8') as f:
        for i in range(count):
            sample = generate_sample(i, filename)
            f.write(json.dumps(sample) + "\n")
    print("Done.")

if __name__ == "__main__":
    # Generate recommended amounts
    create_dataset("data/train2.jsonl", 1000)
    # create_dataset("data/dev.jsonl", 200)