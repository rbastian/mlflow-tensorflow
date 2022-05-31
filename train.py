import sys
import argparse
import tensorflow as tf
import mlflow
import mlflow.tensorflow

# Enable auto-logging to MLflow to capture TensorBoard metrics.
mlflow.tensorflow.autolog(registered_model_name="TensorflowGoogleTutorial")

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default=5, type=int, help="epochs")

def main(argv):
    with mlflow.start_run():
        print("TensorFlow version:", tf.__version__)
        args = parser.parse_args(argv[1:])

        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10)
        ])
       
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    
        model.fit(x_train, y_train, epochs=args.epochs)

        model.evaluate(x_test, y_test, verbose=2)

if __name__ == "__main__":
    main(sys.argv)
