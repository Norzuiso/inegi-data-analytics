from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os

def save_results_to_pdf(results, output_path="resultados_modelos.pdf", cols: list=[], tittle: str = "Resultados del Modelos de Predicción"):
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    margin = 50
    line_height = 14
    y = height - margin

    c.setFont("Helvetica", 12)
    c.drawString(margin, y, tittle)
    y -= 2 * line_height

    for result in results:
        c.setFont("Helvetica-Bold", 11)
        c.drawString(margin, y, f"Modelo: {result['Model']}")
        y -= line_height

        if "Error" in result:
            c.setFont("Helvetica", 10)
            c.drawString(margin, y, f"Error: {result['Error']}")
            y -= 2 * line_height
            continue

        c.setFont("Helvetica", 10)
        c.drawString(margin, y, f"Accuracy: {result['Accuracy']:.4f}")
        y -= line_height
        c.drawString(margin, y, f"F1-Score: {result['F1-Score']:.4f}")
        y -= line_height

        y -= line_height // 2
        c.setFont("Helvetica-Oblique", 10)
        c.drawString(margin, y, "Classification Report:")
        y -= line_height

        for line in result["Classification Report"].split("\n"):
            if y < margin:
                c.showPage()
                y = height - margin
                c.setFont("Helvetica", 10)
            c.drawString(margin + 10, y, line.strip())
            y -= line_height

        y -= 2 * line_height
        if y < margin:
            c.showPage()
            y = height - margin
            c.setFont("Helvetica", 10)
    # Agrega una nueva página para las columnas
    c.showPage()
    y = height - margin
    c.setFont("Helvetica", 12)
    c.drawString(margin, y, "Columnas analizadas")
    y -= 2 * line_height
    for col in cols:
        c.setFont("Helvetica-Bold", 11)
        c.drawString(margin, y, f"Columna : {col}")
        y -= line_height

    c.save()
