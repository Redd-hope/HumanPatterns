import cv2
import dlib
from imutils import face_utils
import imutils


def main():

	IMAGE_PATH = "test8.jpg"

	MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
	age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)',
				'(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
	gender_list = ['Male', 'Female']

	age_net = cv2.dnn.readNetFromCaffe(
		"age_gender_models/deploy_age.prototxt",
		"age_gender_models/age_net.caffemodel")
	gender_net = cv2.dnn.readNetFromCaffe(
		"age_gender_models/deploy_gender.prototxt",
		"age_gender_models/gender_net.caffemodel")

	# Load image
	frame = cv2.imread(IMAGE_PATH)
	frame = imutils.resize(frame, width=600)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Detect faces
	detector = dlib.get_frontal_face_detector()
	rects = detector(gray, 0)

	for i, rect in enumerate(rects):
		(x, y, w, h) = face_utils.rect_to_bb(rect)
		face_img = frame[y:y + h, x:x + w]

		# Predict age and gender
		blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

		gender_net.setInput(blob)
		gender = gender_list[gender_net.forward()[0].argmax()]

		age_net.setInput(blob)
		age = age_list[age_net.forward()[0].argmax()]

		# Draw results
		label = f"{gender}, {age}"
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

		print(f"Face {i + 1}: {gender}, {age}")

	# Show result
	cv2.imshow('Result', frame)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
