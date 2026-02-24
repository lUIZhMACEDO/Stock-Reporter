import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Content, Email, Mail

from main import build_report_html


def send_with_sendgrid(subject: str, html_content: str) -> None:
    api_key = os.getenv("SENDGRID_API_KEY")
    sender = os.getenv("SENDER_EMAIL")
    recipient = os.getenv("RECIPIENT_EMAIL")

    if not all([api_key, sender, recipient]):
        raise ValueError("SENDGRID_API_KEY, SENDER_EMAIL, and RECIPIENT_EMAIL must be set.")

    message = Mail(
        from_email=Email(sender),
        to_emails=Email(recipient),
        subject=subject,
        content=Content("text/html", html_content),
    )

    response = SendGridAPIClient(api_key).send(message)
    if response.status_code >= 400:
        raise RuntimeError(f"SendGrid send failed: status={response.status_code}")


def send_with_gmail(subject: str, html_content: str) -> None:
    sender = os.getenv("SENDER_EMAIL")
    recipient = os.getenv("RECIPIENT_EMAIL")
    app_password = os.getenv("EMAIL_PASSWORD")
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))

    if not all([sender, recipient, app_password]):
        raise ValueError("SENDER_EMAIL, RECIPIENT_EMAIL, and EMAIL_PASSWORD must be set.")

    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = recipient
    message.attach(MIMEText(html_content, "html"))

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(sender, app_password)
        server.sendmail(sender, recipient, message.as_string())


def send_daily_report(test_run: bool = False) -> None:
    subject = "[TEST] Daily S&P 500 Trading Report" if test_run else "Daily S&P 500 Trading Report"
    html_report = build_report_html()
    provider = os.getenv("EMAIL_PROVIDER", "sendgrid").lower()

    if provider == "sendgrid":
        send_with_sendgrid(subject, html_report)
    elif provider == "gmail":
        send_with_gmail(subject, html_report)
    else:
        raise ValueError("EMAIL_PROVIDER must be 'sendgrid' or 'gmail'.")


if __name__ == "__main__":
    send_daily_report(test_run=os.getenv("TEST_RUN", "false").lower() == "true")
