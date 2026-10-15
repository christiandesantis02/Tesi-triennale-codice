
# dopo aver fatto girare su una cella il codice per p_grezza, incollare questo in un'altra cella nella stessa sessione:
def plot_error_vs_time(res, nome_file, titolo_p="p (Pressione)"):
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
    t_vals = res["t"]
    variables = [("u (Velocità x)", "pinn_u", "frozen_u"),
                 ("v (Velocità y)", "pinn_v", "frozen_v"),
                 (titolo_p,         "pinn_p", "frozen_p")]
    for ax, (title, kp, kf) in zip(axs, variables):
        ax.plot(t_vals, res[kp], 'o-', color='tab:blue', linewidth=2, label='PINN')
        ax.plot(t_vals, res[kf], 's--', color='tab:red', linewidth=2, label='Frozen Field')
        ax.set_title(f"Errore Relativo $L^2$ - {title}")
        ax.set_xlabel("Tempo adimensionale $t$")
        ax.set_ylabel("Errore relativo $L^2$")
        ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig(nome_file, dpi=150, bbox_inches="tight")
    plt.show()

# infine, in'un altra cella, incollare:
g = gauge_series(model, t_vals)
for t, (ep, ef) in zip(t_vals, g):
    print(f"t={t:.2f}  PINN {ep:.4e}  frozen {ef:.4e}")

res_g = dict(res)
res_g["pinn_p"]   = [e[0] for e in g]
res_g["frozen_p"] = [e[1] for e in g]
plot_error_vs_time(res_g, "predittore_frozen_field_confronto_Re=500_P_media_nulla.png",
                   titolo_p="p (Pressione a media nulla)")
