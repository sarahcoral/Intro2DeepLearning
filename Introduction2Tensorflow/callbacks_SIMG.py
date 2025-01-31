import keras

# Create custom callback to track and print training progress
class TrainingProgressCallback(keras.callbacks.Callback):
    def on_train_end(self, logs=None):
        if logs is None:
            logs = {}
        print("\nTraining completed!")
        print(f"Final training loss: {logs.get('loss', 'N/A'):.4f}")
        print(f"Final training accuracy: {logs.get('sparse_categorical_accuracy', 'N/A'):.4f}")
        print(f"Final validation loss: {logs.get('val_loss', 'N/A'):.4f}")
        print(f"Final validation accuracy: {logs.get('val_sparse_categorical_accuracy', 'N/A'):.4f}")
        print("""
                _,.---.---.---.--.._ 
           _.-' `--.`---.`---'-. _,`--.._
          /`--._ .'.     `.     `,`-.`-._\\
         ||   \\  `.`---.__`__..-`. ,'`-._/
    _  ,`\\ `-._\\   \\    `.    `_.-`-._,``-.
 ,`   `-_ \\/ `-.`--.\    _\\_.-'\\__.-`-.`-._`.
(_.o> ,--. `._/'--.-`,--`  \\_.-'       \\`-._ \\
 `---'    `._ `---._/__,----`           `-. `-\\
           /_, ,  _..-'                    `-._\\
           \\_, \\/ ._(
            \\_, \\/ ._\\
             `._,\\/ ._\\
               `._// ./`-._
        SIMG      `-._-_-_.-'""")