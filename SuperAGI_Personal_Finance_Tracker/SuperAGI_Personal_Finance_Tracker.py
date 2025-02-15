import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import json

# Generate default sample data
def generate_sample_data():
    dates = [(datetime.now() - timedelta(days=x)).date() for x in range(30)]
    categories = ['Food & Dining', 'Shopping', 'Transportation', 'Bills & Utilities', 'Entertainment']
    
    data = []
    for date in dates:
        # Generate 2-3 transactions per day
        for _ in range(np.random.randint(2, 4)):
            amount = np.random.randint(10, 200)
            category = np.random.choice(categories)
            data.append({
                'date': date,
                'amount': -amount,  # negative for expenses
                'category': category,
                'description': f'Sample {category} transaction',
                'type': 'Expense'
            })
        
        # Add occasional income
        if np.random.random() < 0.1:  # 10% chance of income entry
            data.append({
                'date': date,
                'amount': np.random.randint(1000, 3000),
                'category': 'Income',
                'description': 'Sample Income',
                'type': 'Income'
            })
    
    return pd.DataFrame(data)

# Initialize session state
if 'transactions' not in st.session_state:
    st.session_state.transactions = generate_sample_data()
if 'using_sample_data' not in st.session_state:
    st.session_state.using_sample_data = True

class FinanceAgent:
    def __init__(self, openai_api_key):
        self.llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-3.5-turbo",
            openai_api_key=openai_api_key
        )
        
    def categorize_transaction(self, description):
        """Categorize transaction using LLM"""
        system_prompt = """You are a financial categorization expert. 
        Categorize the transaction into one of these categories:
        - Food & Dining
        - Shopping
        - Transportation
        - Bills & Utilities
        - Entertainment
        - Income
        Return only the category name, nothing else."""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Categorize this transaction: {description}")
        ]
        
        response = self.llm.invoke(messages)
        return response.content.strip()
    
    def get_financial_insights(self, transactions_df):
        """Generate financial insights using LLM"""
        transactions_summary = transactions_df.to_json(orient='records')
        
        system_prompt = """You are a financial advisor. Analyze the transaction data and provide 
        3 key insights about spending patterns and 2 specific recommendations for saving money. 
        Format the response as a JSON with 'insights' and 'recommendations' lists."""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Analyze these transactions: {transactions_summary}")
        ]
        
        response = self.llm.invoke(messages)
        return json.loads(response.content)

def create_spending_chart(transactions_df):
    """Create spending distribution pie chart"""
    monthly_spending = transactions_df[
        transactions_df['type'] == "Expense"
    ].groupby('category')['amount'].sum().abs()
    
    fig = px.pie(
        values=monthly_spending.values,
        names=monthly_spending.index,
        title="Spending Distribution"
    )
    return fig

def create_balance_trend(transactions_df):
    """Create balance trend line chart"""
    daily_balance = transactions_df.groupby('date')['amount'].sum().cumsum()
    fig = px.line(
        x=daily_balance.index,
        y=daily_balance.values,
        title="Balance Over Time",
        labels={'x': 'Date', 'y': 'Balance'}
    )
    return fig

def main():
    st.title("SuperAGI Personal Finance Tracker")
    
    # Sidebar for API key and controls
    with st.sidebar:
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        if not openai_api_key:
            st.warning("Please enter your OpenAI API key to enable AI features")
        
        st.subheader("Add New Transaction")
        date = st.date_input("Date", datetime.now())
        amount = st.number_input("Amount", min_value=0.0, step=0.01)
        description = st.text_input("Description")
        transaction_type = st.selectbox("Type", ["Expense", "Income"])
        
        if st.button("Add Transaction"):
            # Clear sample data if this is the first real transaction
            if st.session_state.using_sample_data:
                st.session_state.transactions = pd.DataFrame(
                    columns=['date', 'amount', 'category', 'description', 'type']
                )
                st.session_state.using_sample_data = False
            
            if openai_api_key:
                agent = FinanceAgent(openai_api_key)
                category = agent.categorize_transaction(description)
            else:
                category = "Uncategorized"
            
            new_transaction = pd.DataFrame([{
                'date': date,
                'amount': amount * (-1 if transaction_type == "Expense" else 1),
                'category': category,
                'description': description,
                'type': transaction_type
            }])
            
            st.session_state.transactions = pd.concat(
                [st.session_state.transactions, new_transaction],
                ignore_index=True
            )
            st.rerun()
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["Dashboard", "Transactions", "AI Insights"])
    
    with tab1:
        st.subheader("Monthly Spending by Category")
        spending_chart = create_spending_chart(st.session_state.transactions)
        st.plotly_chart(spending_chart, use_container_width=True)
        
        st.subheader("Balance Trend")
        balance_chart = create_balance_trend(st.session_state.transactions)
        st.plotly_chart(balance_chart, use_container_width=True)
    
    with tab2:
        st.subheader("Transaction History")
        if st.session_state.using_sample_data:
            st.info("Showing sample data. Add your first transaction to start fresh.")
        
        # Add filters
        categories = ['All'] + list(st.session_state.transactions['category'].unique())
        selected_category = st.selectbox("Filter by Category", categories)
        
        filtered_df = st.session_state.transactions
        if selected_category != 'All':
            filtered_df = filtered_df[filtered_df['category'] == selected_category]
        
        st.dataframe(
            filtered_df.sort_values('date', ascending=False),
            use_container_width=True
        )
    
    with tab3:
        if st.button("Generate AI Insights") and openai_api_key:
            agent = FinanceAgent(openai_api_key)
            with st.spinner("Generating insights..."):
                insights = agent.get_financial_insights(st.session_state.transactions)
                
                st.subheader("💡 Key Insights")
                for insight in insights['insights']:
                    st.write(f"• {insight}")
                
                st.subheader("🎯 Recommendations")
                for recommendation in insights['recommendations']:
                    st.write(f"• {recommendation}")
        elif not openai_api_key:
            st.info("Enter your OpenAI API key in the sidebar to generate AI insights")

if __name__ == "__main__":
    main()
