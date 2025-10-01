from typing import Dict
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


def generate_pdf_report(path: str, title: str, metrics: Dict[str, float], fairness: Dict[str, float]) -> None:
	c = canvas.Canvas(path, pagesize=A4)
	width, height = A4
	c.setFont("Helvetica-Bold", 16)
	c.drawString(72, height - 72, title)
	c.setFont("Helvetica", 12)
	y = height - 110
	c.drawString(72, y, "Model Metrics:")
	y -= 20
	for k, v in metrics.items():
		c.drawString(90, y, f"{k}: {v:.4f}")
		y -= 16
	y -= 10
	c.drawString(72, y, "Fairness Metrics:")
	y -= 20
	for k, v in fairness.items():
		c.drawString(90, y, f"{k}: {v:.4f}")
		y -= 16
	c.showPage()
	c.save() 