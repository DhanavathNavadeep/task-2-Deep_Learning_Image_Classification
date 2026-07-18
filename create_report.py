from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

doc = SimpleDocTemplate("report.pdf")
styles = getSampleStyleSheet()

story = [
	Paragraph("Deep Learning Image Classification Report", styles["Title"]),
	Paragraph("Project Title", styles["Heading2"]),
	Paragraph("Cats vs Dogs Image Classification using TensorFlow/Keras", styles["BodyText"]),
	Paragraph("Objective", styles["Heading2"]),
	Paragraph("Develop a CNN model to classify cats and dogs.", styles["BodyText"]),
	Paragraph("Results", styles["Heading2"]),
	Paragraph("Validation Accuracy: 99.21%", styles["BodyText"]),
	Paragraph("Validation Loss: 0.0224", styles["BodyText"]),
]

doc.build(story)

print("report.pdf created successfully.")