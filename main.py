import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from ACO import AntColony


def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2.0) ** 2 + \
        np.cos(phi1) * np.cos(phi2) * \
        np.sin(delta_lambda / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


def plot_path(coords, path, iteration, distance):
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())  # klasyczne rzutowanie geograficzne
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.set_title(f'Iteration: {iteration}, Distance: {distance:.2f} km')

    # Wyciągnięcie punktów w trasie
    lats = [coords[i][0] for i in path] + [coords[path[0]][0]]
    lons = [coords[i][1] for i in path] + [coords[path[0]][1]]

    ax.plot(lons, lats, color='blue', marker='o', transform=ccrs.Geodetic())  # rysowanie trasy po kuli
    plt.show()


def animate(i, coords, history, line, ax, ax_cost):
    path, dist = history[i]
    path_coords = coords[path + [path[0]]]

    lats = path_coords[:, 0]
    lons = path_coords[:, 1]

    line.set_data(lons, lats)
    ax.set_title(f'Iteration {i + 1}, Distance: {dist:.2f} km')

    # Update cost plot
    ax_cost.clear()
    ax_cost.plot(range(i + 1), [cost for _, cost in history[:i + 1]], color='green')
    ax_cost.set_title('Cost Function')
    ax_cost.set_xlabel('Iteration')
    ax_cost.set_ylabel('Cost')
    ax_cost.grid(True)

    return line, ax, ax_cost

    ax.set_title(f'Iteration {i + 1}, Distance: {dist:.2f} km')
    return scat, line, ax_cost


def main():
    df = pd.read_csv("airports.csv")
    df_usa = df[df['country'] == 'USA']
    df_largest = df_usa.sort_values('airport').groupby('state').first().reset_index()
    df_largest = df_largest[['airport', 'city', 'state', 'country', 'lat', 'long']]

    coords = df_largest[['lat', 'long']].to_numpy()
    n = len(coords)
    dist_matrix = np.array([[haversine(*coords[i], *coords[j]) for j in range(n)] for i in range(n)])
    print(dist_matrix)

    aco = AntColony(dist_matrix, n_ants=20, n_best=10, n_iterations=100, decay=0.4, alpha=1.0, beta=4.0,patience=30) #20,10,100,0.6,1.1,2.0
    aco.run()

    # Create two subplots: one for path and one for cost
    fig, (ax, ax_cost) = plt.subplots(1, 2, figsize=(15, 6))

    scat = ax.scatter(coords[:, 1], coords[:, 0], c='red')
    line, = ax.plot([], [], 'b-', lw=2)

    ani = FuncAnimation(fig, animate, frames=len(aco.history), fargs=(coords, aco.history, line, ax, ax_cost),
                        interval=200, repeat=False)
    plt.show()
    ani.save('aco_animation.gif', writer='Pillow', fps=10)

if __name__ == "__main__":
    main()
