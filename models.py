from flask_login import UserMixin
from extensions import db

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(150))
    is_admin = db.Column(db.Boolean, default=False)

class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    Amount = db.Column(db.Float)
    MerchantCategory = db.Column(db.String(100))
    TransactionType = db.Column(db.String(100))
    Latitude = db.Column(db.Float)
    Longitude = db.Column(db.Float)
    AvgTransactionAmount = db.Column(db.Float)
    TransactionFrequency = db.Column(db.String(100))
    UnusualLocation = db.Column(db.String(10))
    UnusualAmount = db.Column(db.String(10))
    NewDevice = db.Column(db.String(10))
    FailedAttempts = db.Column(db.Integer)
    BankName = db.Column(db.String(100))
    result = db.Column(db.String(10))
