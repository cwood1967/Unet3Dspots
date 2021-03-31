import os
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.message import EmailMessage

import smtplib
import socket

logging.basicConfig(format='%(asctime)s %(message)s',
                    filename='maillogs/email.log',
                    level=logging.INFO)

def message(subject, text):
    msg = EmailMessage()
    msg['Subject'] = subject
    msg.set_content(text)

    return msg

def send(msg, to, server='smtp.office365.com', port=587):
    try:
        smtp = smtplib.SMTP(server, port)
        smtp.ehlo()
        smtp.starttls()

        email = 'simr_micro@outlook.com'
        passwd = 'Micro!23'
        smtp.login(email, passwd)
        smtp.sendmail(email, to, msg.as_string())
        smtp.quit()

    except socket.gaierror:
        logging.warning("Can't send email, network error")

def notify_error(error, filepath, to_list):
    
    to = ",".join(to_list)
    msg = MIMEMultipart()
    msg['Subject'] = 'Error Notification'
    msg["From"] = "simr_micro@outlook.com"
    msg["To"] = to
    text = f"""\
    An error has occured during acquisition:
    {error}
    in the file
    {filepath}
    """
    body = MIMEText(text)
    msg.attach(body)
    #msg.set_content(text)
    
    send(msg, to.split(','))
    logging.info(f'{msg.as_string()}\n-------')

def test():
    msg = message("Oh no!", "Something went wrong")
    send(msg, "cwood1967@gmail.com")
