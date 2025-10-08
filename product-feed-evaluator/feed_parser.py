"""
XML feed parsing and data cleaning logic for the Product Feed Evaluation Agent.
"""

import xml.etree.ElementTree as ET
import re
import urllib.parse
from typing import List, Dict, Any, Optional
import pandas as pd


class FeedParser:
    """Handles parsing and cleaning of Google Shopping XML feeds."""
    
    def __init__(self):
        self.required_fields = ['link', 'title', 'description', 'image_link']
        self.recommended_fields = [
            'brand', 'gtin', 'mpn', 'price', 'availability', 
            'color', 'size', 'material', 'product_type', 'item_group_id'
        ]
    
    def parse_xml_feed(self, xml_content: str) -> List[Dict[str, Any]]:
        """
        Parse XML feed content and extract product data.
        
        Args:
            xml_content: Raw XML content as string
            
        Returns:
            List of product dictionaries
        """
        try:
            root = ET.fromstring(xml_content)
            products = []
            
            # Handle both RSS and Atom feeds
            items = root.findall('.//item') or root.findall('.//entry')
            
            for item in items:
                product = self._extract_product_data(item)
                if product and product.get('link'):
                    products.append(product)
            
            return products
            
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML format: {e}")
    
    def _extract_product_data(self, item: ET.Element) -> Dict[str, Any]:
        """Extract product data from a single item element."""
        product = {}
        
        # Map common field names
        field_mapping = {
            'id': 'id',
            'title': 'title',
            'description': 'description',
            'link': 'link',
            'image_link': 'image_link',
            'brand': 'brand',
            'gtin': 'gtin',
            'mpn': 'mpn',
            'price': 'price',
            'availability': 'availability',
            'color': 'color',
            'size': 'size',
            'material': 'material',
            'product_type': 'product_type',
            'item_group_id': 'item_group_id'
        }
        
        for xml_field, our_field in field_mapping.items():
            element = item.find(xml_field)
            if element is not None and element.text:
                product[our_field] = element.text.strip()
        
        return product
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing HTML tags, excess whitespace, and tracking parameters.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove tracking parameters from URLs
        if text.startswith('http'):
            parsed = urllib.parse.urlparse(text)
            if parsed.query:
                # Remove common tracking parameters
                query_params = urllib.parse.parse_qs(parsed.query)
                tracking_params = ['utm_source', 'utm_medium', 'utm_campaign', 'utm_content', 'utm_term', 'fbclid', 'gclid']
                for param in tracking_params:
                    query_params.pop(param, None)
                
                # Rebuild URL without tracking parameters
                new_query = urllib.parse.urlencode(query_params, doseq=True)
                text = urllib.parse.urlunparse((
                    parsed.scheme, parsed.netloc, parsed.path, 
                    parsed.params, new_query, parsed.fragment
                ))
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def deduplicate_products(self, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate products, preferring parent entries for item groups.
        
        Args:
            products: List of product dictionaries
            
        Returns:
            Deduplicated list of products
        """
        seen_urls = set()
        deduplicated = []
        
        # First pass: collect item_group_id mappings
        group_parents = {}
        for product in products:
            if product.get('item_group_id') and not product.get('link') in seen_urls:
                group_id = product['item_group_id']
                if group_id not in group_parents:
                    group_parents[group_id] = product
        
        # Second pass: add products, preferring parents
        for product in products:
            url = product.get('link')
            if not url or url in seen_urls:
                continue
                
            group_id = product.get('item_group_id')
            if group_id and group_id in group_parents:
                # Prefer parent entry
                if product == group_parents[group_id]:
                    deduplicated.append(product)
                    seen_urls.add(url)
            else:
                # No group or this is the parent
                deduplicated.append(product)
                seen_urls.add(url)
        
        return deduplicated
    
    def create_product_context(self, product: Dict[str, Any], selected_fields: List[str]) -> str:
        """
        Create a combined context string from selected fields.
        
        Args:
            product: Product dictionary
            selected_fields: List of field names to include
            
        Returns:
            Combined context string
        """
        context_parts = []
        
        for field in selected_fields:
            if field in product and product[field]:
                cleaned_value = self.clean_text(str(product[field]))
                if cleaned_value:
                    context_parts.append(f"{field.title()}: {cleaned_value}")
        
        return " | ".join(context_parts)
    
    def check_quality_issues(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check for quality issues in product data.
        
        Args:
            product: Product dictionary
            
        Returns:
            Dictionary of quality issues found
        """
        issues = {}
        
        # Check missing essential fields
        missing_essential = []
        for field in self.required_fields:
            if not product.get(field):
                missing_essential.append(field)
        
        if missing_essential:
            issues['missing_essential_fields'] = ', '.join(missing_essential)
        
        # Check title length
        title = product.get('title', '')
        if len(title) < 20:
            issues['title_too_short'] = True
        elif len(title) > 150:
            issues['title_too_long'] = True
        
        # Check description length
        description = product.get('description', '')
        if len(description) < 60:
            issues['description_too_short'] = True
        elif len(description) > 5000:
            issues['description_too_long'] = True
        
        # Check for ALL CAPS or excessive emojis
        if title.isupper() or len(re.findall(r'[^\w\s]', title)) > len(title) * 0.3:
            issues['poor_title_quality'] = True
        
        # Check image link validity
        image_link = product.get('image_link', '')
        if image_link and not self._is_valid_image_url(image_link):
            issues['invalid_image_link'] = True
        
        return issues
    
    def _is_valid_image_url(self, url: str) -> bool:
        """Check if URL appears to be a valid image link."""
        if not url.startswith('http'):
            return False
        
        # Check for common image extensions
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg']
        return any(url.lower().endswith(ext) for ext in image_extensions)
    
    def process_feed(self, xml_content: str, selected_fields: List[str]) -> pd.DataFrame:
        """
        Complete feed processing pipeline.
        
        Args:
            xml_content: Raw XML content
            selected_fields: Fields to include in context
            
        Returns:
            DataFrame with processed products
        """
        # Parse XML
        products = self.parse_xml_feed(xml_content)
        
        # Clean and deduplicate
        cleaned_products = []
        for product in products:
            cleaned_product = {}
            for key, value in product.items():
                if isinstance(value, str):
                    cleaned_product[key] = self.clean_text(value)
                else:
                    cleaned_product[key] = value
            cleaned_products.append(cleaned_product)
        
        deduplicated_products = self.deduplicate_products(cleaned_products)
        
        # Create context and check quality
        processed_products = []
        for product in deduplicated_products:
            context = self.create_product_context(product, selected_fields)
            quality_issues = self.check_quality_issues(product)
            
            processed_product = {
                'product_url': product.get('link', ''),
                'name_plus_context': context,
                'quality_issues': quality_issues
            }
            processed_product.update(product)
            processed_products.append(processed_product)
        
        return pd.DataFrame(processed_products)