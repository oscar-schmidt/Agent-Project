#!/usr/bin/env python3
"""
Simple Flask server for adding reviews manually
"""
from flask import Flask, request, jsonify, send_file
import sys
import os


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import insert_new_review, get_next_review_id

app = Flask(__name__)

@app.route('/')
def index():
    """Serve the review form"""
    return send_file('review_form.html')

@app.route('/next-id')
def next_id():
    """Get the next review ID"""
    try:
        next_id = get_next_review_id()
        return jsonify({'next_id': next_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/add-review', methods=['POST'])
def add_review():
    """add a new review to the database"""
    try:
        data = request.json

        #check required fields
        required_fields = ['review', 'username', 'email', 'reviewer_name', 'rating']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({'success': False, 'error': f'Missing field: {field}'}), 400

        #insert review
        review_id = insert_new_review(
            review_text=data['review'],
            username=data['username'],
            email=data['email'],
            reviewer_name=data['reviewer_name'],
            rating=int(data['rating'])
        )

        #get next ID for the form
        next_id = get_next_review_id()

        return jsonify({
            'success': True,
            'review_id': review_id,
            'next_id': next_id
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    #print("Starting review entry server...")
    print("Open http://localhost:5000 to add reviews")
    app.run(debug=True, port=5000)