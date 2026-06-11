"""
Report generation and email utilities
"""
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os
from datetime import datetime
import json

class ReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.custom_styles = self._create_custom_styles()
    
    def _create_custom_styles(self):
        """Create custom paragraph styles"""
        styles = {}
        
        styles['CustomTitle'] = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#8b5cf6'),
            alignment=1  # Center alignment
        )
        
        styles['CustomHeading'] = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.HexColor('#ec4899')
        )
        
        return styles
    
    def generate_prediction_report(self, customer_data, predictions, model_performance, output_path='reports/churn_report.pdf'):
        """Generate comprehensive PDF report"""
        
        # Create reports directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        story = []
        
        # Title
        title = Paragraph("Churn Prediction Analysis Report", self.custom_styles['CustomTitle'])
        story.append(title)
        story.append(Spacer(1, 20))
        
        # Report metadata
        report_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        metadata = Paragraph(f"<b>Generated:</b> {report_date}<br/><b>Report Type:</b> Customer Churn Analysis", 
                            self.styles['Normal'])
        story.append(metadata)
        story.append(Spacer(1, 20))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", self.custom_styles['CustomHeading']))
        
        if predictions:
            high_risk_count = sum(1 for p in predictions.values() if p['probability'] > 0.7)
            total_customers = len(predictions)
            avg_risk = sum(p['probability'] for p in predictions.values()) / total_customers
            
            summary_text = f"""
            This report analyzes churn risk for {total_customers} customers using multiple machine learning models.
            <br/><br/>
            <b>Key Findings:</b><br/>
            • {high_risk_count} customers ({high_risk_count/total_customers*100:.1f}%) are at high risk of churning<br/>
            • Average churn probability across all customers: {avg_risk:.1%}<br/>
            • Recommended immediate action for customers with >70% churn probability
            """
        else:
            summary_text = "No prediction data available for analysis."
        
        story.append(Paragraph(summary_text, self.styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Model Performance Section
        if model_performance:
            story.append(Paragraph("Model Performance Analysis", self.custom_styles['CustomHeading']))
            
            # Create performance table
            perf_data = [['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']]
            for model_name, metrics in model_performance.items():
                if 'error' not in metrics:
                    perf_data.append([
                        model_name,
                        f"{metrics.get('accuracy', 0):.3f}",
                        f"{metrics.get('precision', 0):.3f}",
                        f"{metrics.get('recall', 0):.3f}",
                        f"{metrics.get('f1', 0):.3f}",
                        f"{metrics.get('auc', 0):.3f}"
                    ])
            
            perf_table = Table(perf_data)
            perf_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#8b5cf6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(perf_table)
            story.append(Spacer(1, 20))
        
        # Customer Risk Analysis
        if predictions:
            story.append(Paragraph("Customer Risk Analysis", self.custom_styles['CustomHeading']))
            
            # Risk distribution
            risk_levels = {'Low (0-20%)': 0, 'Moderate (20-50%)': 0, 'High (50-75%)': 0, 'Critical (75%+)': 0}
            
            for pred in predictions.values():
                prob = pred['probability']
                if prob < 0.2:
                    risk_levels['Low (0-20%)'] += 1
                elif prob < 0.5:
                    risk_levels['Moderate (20-50%)'] += 1
                elif prob < 0.75:
                    risk_levels['High (50-75%)'] += 1
                else:
                    risk_levels['Critical (75%+)'] += 1
            
            risk_data = [['Risk Level', 'Customer Count', 'Percentage']]
            for level, count in risk_levels.items():
                percentage = (count / total_customers) * 100 if total_customers > 0 else 0
                risk_data.append([level, str(count), f"{percentage:.1f}%"])
            
            risk_table = Table(risk_data)
            risk_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ec4899')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(risk_table)
            story.append(Spacer(1, 20))
        
        # Recommendations
        story.append(Paragraph("Recommendations", self.custom_styles['CustomHeading']))
        recommendations = """
        <b>Immediate Actions:</b><br/>
        1. Contact high-risk customers (>70% churn probability) within 48 hours<br/>
        2. Offer personalized retention incentives to moderate-risk customers<br/>
        3. Implement proactive engagement campaigns for at-risk segments<br/><br/>
        
        <b>Long-term Strategies:</b><br/>
        1. Enhance customer experience based on key churn indicators<br/>
        2. Develop predictive intervention workflows<br/>
        3. Regular model retraining with new customer data<br/>
        4. A/B testing of retention strategies
        """
        
        story.append(Paragraph(recommendations, self.styles['Normal']))
        
        # Build PDF
        doc.build(story)
        return output_path
    
    def send_email_alert(self, recipient_email, subject, body, attachment_path=None, 
                        smtp_server='smtp.gmail.com', smtp_port=587, sender_email=None, sender_password=None):
        """Send email alert with optional attachment"""
        
        if not sender_email or not sender_password:
            print("Email credentials not configured")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = recipient_email
            msg['Subject'] = subject
            
            # Add body
            msg.attach(MIMEText(body, 'html'))
            
            # Add attachment if provided
            if attachment_path and os.path.exists(attachment_path):
                with open(attachment_path, "rb") as attachment:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment.read())
                
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {os.path.basename(attachment_path)}'
                )
                msg.attach(part)
            
            # Send email
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(sender_email, sender_password)
            text = msg.as_string()
            server.sendmail(sender_email, recipient_email, text)
            server.quit()
            
            return True
            
        except Exception as e:
            print(f"Email sending failed: {e}")
            return False
    
    def generate_high_risk_alert(self, high_risk_customers, recipient_email):
        """Generate and send high-risk customer alert"""
        
        if not high_risk_customers:
            return False
        
        subject = f"🚨 High Risk Customer Alert - {len(high_risk_customers)} Customers Need Immediate Attention"
        
        body = f"""
        <html>
        <body>
        <h2 style="color: #ef4444;">High Risk Customer Alert</h2>
        <p>The following customers have been identified as high risk for churn (>70% probability):</p>
        
        <table border="1" style="border-collapse: collapse; width: 100%;">
        <tr style="background-color: #f2f2f2;">
            <th>Customer ID</th>
            <th>Churn Probability</th>
            <th>Risk Level</th>
            <th>Recommended Action</th>
        </tr>
        """
        
        for customer_id, data in high_risk_customers.items():
            prob = data['probability']
            if prob > 0.9:
                risk_level = "CRITICAL"
                action = "Immediate call + retention offer"
            elif prob > 0.8:
                risk_level = "VERY HIGH"
                action = "Priority outreach within 24h"
            else:
                risk_level = "HIGH"
                action = "Contact within 48h"
            
            body += f"""
            <tr>
                <td>{customer_id}</td>
                <td>{prob:.1%}</td>
                <td style="color: #ef4444;"><b>{risk_level}</b></td>
                <td>{action}</td>
            </tr>
            """
        
        body += """
        </table>
        
        <p><b>Next Steps:</b></p>
        <ul>
            <li>Review customer history and preferences</li>
            <li>Prepare personalized retention offers</li>
            <li>Schedule follow-up actions</li>
            <li>Update CRM with intervention results</li>
        </ul>
        
        <p>This alert was generated automatically by the Churn Prediction System.</p>
        </body>
        </html>
        """
        
        return self.send_email_alert(recipient_email, subject, body)

# Global report generator
report_generator = ReportGenerator()