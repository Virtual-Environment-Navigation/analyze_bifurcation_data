import matplotlib.pyplot as plt

def plot_models(x_data, y_data, unimodal_model, bimodal_model):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.plot(x_data, y_data, label='unimodal model', marker='o')
    ax1.plot(x_data, unimodal_model, label='unimodal model', marker='x')
    ax1.legend()

    ax2.plot(x_data, y_data, label='bimodal model', marker='o')
    ax2.plot(x_data, bimodal_model, label='bimodal model', marker='x')
    ax2.legend()

    plt.show()
