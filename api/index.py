from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
import os
import joblib
import pandas as pd

from helpers import preprocess_and_encode
from models import User, Transaction
from extensions import db
from flask_migrate import Migrate

app = Flask(__name__)
app.secret_key = 'secret'

# PostgreSQL connection (should be set in Vercel Environment Variables)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db.init_app(app)
migrate = Migrate(app, db)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Create upload folder if not exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load the trained model
model = joblib.load('xgb_model.pkl')

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and user.password == request.form['password']:
            login_user(user)
            return redirect(url_for('admin_dashboard' if user.is_admin else 'dashboard'))
        flash('Invalid credentials')
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        if User.query.filter_by(username=request.form['username']).first():
            flash('Username already exists')
            return redirect(url_for('register'))
        user = User(username=request.form['username'], password=request.form['password'])
        db.session.add(user)
        db.session.commit()
        flash('Registered successfully')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', username=current_user.username)

@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    if not current_user.is_admin:
        flash("Access Denied")
        return redirect(url_for('dashboard'))
    return render_template('admin_dashboard.html', username=current_user.username)

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or not file.filename.endswith('.csv'):
            flash('Upload a valid CSV file')
            return redirect(request.url)

        path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(path)

        try:
            df = pd.read_csv(path)
            required_cols = ['Amount', 'MerchantCategory', 'TransactionType', 'Latitude', 'Longitude',
                             'AvgTransactionAmount', 'TransactionFrequency', 'UnusualLocation',
                             'UnusualAmount', 'NewDevice', 'FailedAttempts', 'BankName']
            if not all(col in df.columns for col in required_cols):
                flash('Missing required columns in CSV')
                return redirect(request.url)

            for _, row in df.iterrows():
                t = Transaction(
                    user_id=current_user.id,
                    Amount=row['Amount'],
                    MerchantCategory=row['MerchantCategory'],
                    TransactionType=row['TransactionType'],
                    Latitude=row['Latitude'],
                    Longitude=row['Longitude'],
                    AvgTransactionAmount=row['AvgTransactionAmount'],
                    TransactionFrequency=row['TransactionFrequency'],
                    UnusualLocation=row['UnusualLocation'],
                    UnusualAmount=row['UnusualAmount'],
                    NewDevice=row['NewDevice'],
                    FailedAttempts=row['FailedAttempts'],
                    BankName=row['BankName']
                )
                db.session.add(t)
            db.session.commit()
            flash('Uploaded successfully')
            return redirect(url_for('predict_all'))
        except Exception as e:
            flash(f'Error: {e}')
    return render_template('upload.html')

@app.route('/predict_all')
@login_required
def predict_all():
    transactions = Transaction.query.filter_by(user_id=current_user.id).all()
    if not transactions:
        flash('No transactions found')
        return redirect(url_for('upload'))

    df = pd.DataFrame([{
        'Amount': t.Amount,
        'MerchantCategory': t.MerchantCategory,
        'TransactionType': t.TransactionType,
        'Latitude': t.Latitude,
        'Longitude': t.Longitude,
        'AvgTransactionAmount': t.AvgTransactionAmount,
        'TransactionFrequency': t.TransactionFrequency,
        'UnusualLocation': t.UnusualLocation,
        'UnusualAmount': t.UnusualAmount,
        'NewDevice': t.NewDevice,
        'FailedAttempts': t.FailedAttempts,
        'BankName': t.BankName
    } for t in transactions])

    X = preprocess_and_encode(df)
    predictions = model.predict(X)

    for t, pred in zip(transactions, predictions):
        t.result = 'Fraud' if pred == 1 else 'Legit'
    db.session.commit()

    return render_template('transactions.html', transactions=transactions)

@app.route('/admin/view_all')
@login_required
def view_all_transactions():
    if not current_user.is_admin:
        flash('Unauthorized')
        return redirect(url_for('dashboard'))
    transactions = Transaction.query.all()
    return render_template('admin_transactions.html', transactions=transactions)

@app.route('/admin/delete/<int:transaction_id>')
@login_required
def delete_transaction(transaction_id):
    if not current_user.is_admin:
        flash('Unauthorized')
        return redirect(url_for('dashboard'))
    txn = Transaction.query.get(transaction_id)
    if txn:
        db.session.delete(txn)
        db.session.commit()
        flash('Transaction deleted')
    else:
        flash('Transaction not found')
    return redirect(url_for('view_all_transactions'))

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))
