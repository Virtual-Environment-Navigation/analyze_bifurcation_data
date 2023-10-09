import matplotlib.pyplot as plt

def plot_models(x_data, y_data, unimodal_model, bimodal_model):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.hist(y_data, bins=30, density=True, alpha=0.5, label='data')
    ax1.plot(x_data, unimodal_model, label='unimodal model')
    ax1.legend()

    ax2.hist(y_data, bins=30, density=True, alpha=0.5, label='data')
    ax2.plot(x_data, bimodal_model, label='bimodal model')
    ax2.legend()

    plt.show()
