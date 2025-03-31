import re
import json
import os
import pickle
import PyPDF2
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

class PoliceReportNER:
    """
    A class for extracting named entities from police reports
    """
    def __init__(self, model_name="dslim/bert-base-NER"):
        """
        Initialize the NER model
        
        Args:
            model_name: The name of the pre-trained model to use
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.ner = pipeline("ner", model=self.model, tokenizer=self.tokenizer)
        
        # Define regex patterns for structured fields
        self.patterns = {
            "report_number": r"Report Number:\s*([^\n]+)",
            "date_time": r"Date & Time:\s*([^\n]+)",
            "reporting_officer": r"Reporting Officer:\s*([^\n]+)",
            "incident_location": r"Incident Location:\s*([^\n]+)",
            "coordinates": r"Coordinates:\s*([^\n]+)",
            "detailed_description": r"Detailed Description:(.*?)(?=Police District:|$)",
            "police_district": r"Police District:\s*([^\n]+)",
            "resolution": r"Resolution:\s*([^\n]+)",
            "suspect_description": r"Suspect Description:(.*?)(?=Victim Information:|$)",
            "victim_information": r"Victim Information:(.*?)(?=$)"
        }
    
    def extract_entities(self, report_text):
        """
        Extract named entities from a police report
        
        Args:
            report_text: The text of the police report
            
        Returns:
            A dictionary containing the extracted entities
        """
        # Initialize result dictionary
        result = {}
        
        # Extract structured fields using regex patterns
        for field, pattern in self.patterns.items():
            match = re.search(pattern, report_text, re.DOTALL)
            if match:
                result[field] = match.group(1).strip()
        
        # Process coordinates if present
        if "coordinates" in result:
            coord_str = result["coordinates"]
            # Clean up coordinates and convert to dict with lat/lng
            coord_str = coord_str.replace('(', '').replace(')', '')
            try:
                lat, lng = coord_str.split(',')
                result["coordinates"] = {
                    "latitude": float(lat.strip()),
                    "longitude": float(lng.strip())
                }
            except:
                # Keep as string if parsing fails
                pass
        
        # For less structured parts of the text, use NER
        if "detailed_description" in result:
            description_text = result["detailed_description"]
            
            # Get NER results
            ner_results = self.ner(description_text)
            
            # Group entities by type
            crime_types = []
            items_stolen = []
            
            # Process NER results
            for entity in ner_results:
                if entity['entity'] in ['B-MISC', 'I-MISC']:
                    if entity['word'].lower() in ['theft', 'stolen', 'robbery', 'burglary', 'assault']:
                        crime_types.append(entity['word'])
                elif entity['entity'] in ['B-ORG', 'I-ORG']:
                    items_stolen.append(entity['word'])
            
            # Add extracted entities to result
            if crime_types:
                result["crime_type"] = " ".join(crime_types)
            
            if items_stolen:
                result["items_stolen"] = items_stolen
        
        return result
    
    def extract_entities_from_pdf(self, pdf_path):
        """
        Extract named entities from a PDF police report
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            A dictionary containing the extracted entities
        """
        # Extract text from PDF
        text = self.extract_text_from_pdf(pdf_path)
        
        # Extract entities from the text
        return self.extract_entities(text)
    
    def extract_text_from_pdf(self, pdf_path):
        """
        Extract text from a PDF file
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            The extracted text
        """
        text = ""
        
        try:
            # Open the PDF file
            with open(pdf_path, 'rb') as file:
                # Create a PDF reader object
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract text from each page
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
        
        return text
    
    def save_model(self, directory="police_ner_model"):
        """
        Save the model and tokenizer to a directory
        
        Args:
            directory: The directory to save the model to
        """
        # Create the directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Save the model and tokenizer
        self.model.save_pretrained(os.path.join(directory, "model"))
        self.tokenizer.save_pretrained(os.path.join(directory, "tokenizer"))
        
        # Save the patterns
        with open(os.path.join(directory, "patterns.pkl"), "wb") as f:
            pickle.dump(self.patterns, f)
        
        print(f"Model saved to {directory}")
    
    @classmethod
    def load_model(cls, directory="police_ner_model"):
        """
        Load a saved model from a directory
        
        Args:
            directory: The directory to load the model from
            
        Returns:
            A PoliceReportNER instance
        """
        # Create a new instance
        instance = cls.__new__(cls)
        
        # Load the model and tokenizer
        instance.model = AutoModelForTokenClassification.from_pretrained(os.path.join(directory, "model"))
        instance.tokenizer = AutoTokenizer.from_pretrained(os.path.join(directory, "tokenizer"))
        instance.ner = pipeline("ner", model=instance.model, tokenizer=instance.tokenizer)
        
        # Load the patterns
        with open(os.path.join(directory, "patterns.pkl"), "rb") as f:
            instance.patterns = pickle.load(f)
        
        return instance

# Example usage for testing with a PDF file
def test_with_pdf(pdf_path, model_directory="police_ner_model"):
    """
    Test the NER model on a PDF file
    
    Args:
        pdf_path: Path to the PDF file
        model_directory: Directory containing the saved model
    """
    # Load the model
    try:
        ner = PoliceReportNER.load_model(model_directory)
        print(f"Successfully loaded model from {model_directory}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Initializing a new model instead...")
        ner = PoliceReportNER()
        # Save the model for future use
        ner.save_model(model_directory)
    
    try:
        # Process the PDF file
        extracted_info = ner.extract_entities_from_pdf(pdf_path)
        
        # Output as JSON
        print("Extracted Information:")
        print(json.dumps(extracted_info, indent=2))
        
        # Save the results to a JSON file
        output_file = f"{os.path.splitext(pdf_path)[0]}_extracted.json"
        with open(output_file, "w") as f:
            json.dump(extracted_info, f, indent=2)
        
        print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Error processing PDF: {e}")

if __name__ == "__main__":
    # Test with your PDF file
    pdf_path = "police_crime_report_10.pdf"
    test_with_pdf(pdf_path)
