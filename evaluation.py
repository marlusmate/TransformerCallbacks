import pandas as pd
import matplotlib.pyplot as plt

# Load
csv_dir_vivit = 'runs_vivit.csv'
csv_dir_vswin = 'runs_vswin.csv'
df_vivit = pd.read_csv(csv_dir_vivit, sep=',', header=0).dropna(subset='auc')
df_vswin = pd.read_csv(csv_dir_vswin, sep=',', header=0).dropna(subset='auc').sort_values('n_inst_percentage')

df_vivit_pretrained = df_vivit[df_vivit["transfer_learning"]==True]
df_vivit_scratch = df_vivit[df_vivit["transfer_learning"]==False]
df_vswin_pretrained = df_vswin[df_vswin["transfer_learning"]==True]
df_vswin_scratch = df_vswin[df_vswin["transfer_learning"]==False]

# Plot
#Vivit
#AUC
f1, ax1 = plt.subplots()
ax1.plot(df_vivit_pretrained["n_inst_percentage"], df_vivit_pretrained["auc"], 'x--')
ax1.plot(df_vivit_scratch["n_inst_percentage"], df_vivit_scratch["auc"], 'o--')
ax1.set(title="vivit-tiny-patch16-224", xlabel="Percentage Used Training Labels", ylabel="AUC", xlim=[0,105], ylim=[0,1])
ax1.grid()
f1.legend(["Pretrained", "Scratch"])
f1.savefig("FinalEval/auc_vivit")
#f1.show()


#ACC
f2, ax2 = plt.subplots()
ax2.plot(df_vivit_pretrained["n_inst_percentage"], df_vivit_pretrained["Accuracy"], 'x--')
ax2.plot(df_vivit_scratch["n_inst_percentage"], df_vswin_scratch["Accuracy"], 'o--')
ax2.set(title="vswin-tiny-patch4-window7-224", xlabel="Percentage Used Training Labels", ylabel="Accuracy", xlim=[0,105], ylim=[0,1])
ax2.grid()
f2.legend(["Pretrained", "Scratch"])
f2.savefig("FinalEval/acc_vivit")
#f2.show()

#F1
f3, ax3 = plt.subplots()
ax3.plot(df_vivit_pretrained["n_inst_percentage"], df_vivit_pretrained["f1"], 'x--')
ax3.plot(df_vivit_scratch["n_inst_percentage"], df_vivit_scratch["f1"], 'o--')
ax3.set(title="vivit-tiny-patch16-224", xlabel="Percentage Used Training Labels", ylabel="F1", xlim=[0,105], ylim=[0,1])
ax3.grid()
f3.legend(["Pretrained", "Scratch"])
f3.savefig("FinalEval/f1_vivit")
#f3.show()


#VSWIN
#AUC
f4, ax4 = plt.subplots()
ax4.plot(df_vswin_pretrained["n_inst_percentage"], df_vswin_pretrained["auc"], 'x--')
ax4.plot(df_vswin_scratch["n_inst_percentage"], df_vswin_scratch["auc"], 'o--')
ax4.set(title="vswin-tiny-patch4-window7-224", xlabel="Percentage Used Training Labels", ylabel="AUC", xlim=[0,105], ylim=[0,1])
ax4.grid()
f4.legend(["Pretrained", "Scratch"])
f4.savefig("FinalEval/auc_vswin")
f4.show()


#ACC
f5, ax5 = plt.subplots()
ax5.plot(df_vswin_pretrained["n_inst_percentage"], df_vswin_pretrained["Accuracy"], 'x--')
ax5.plot(df_vswin_scratch["n_inst_percentage"], df_vswin_scratch["Accuracy"], 'o--')
ax5.set(title="vswin-tiny-patch4-window7-224", xlabel="Percentage Used Training Labels", ylabel="Accuracy", xlim=[0,105], ylim=[0,1])
ax5.grid()
f5.legend(["Pretrained", "Scratch"])
f5.savefig("FinalEval/acc_vswin")
f5.show()

#F1
f6, ax6 = plt.subplots()
ax6.plot(df_vswin_pretrained["n_inst_percentage"], df_vswin_pretrained["f1"], 'x--')
ax6.plot(df_vswin_scratch["n_inst_percentage"], df_vswin_scratch["f1"], 'o--')
ax6.set(title="vswin-tiny-patch4-window7-224", xlabel="Percentage Used Training Labels", ylabel="F1", xlim=[0,105], ylim=[0,1])
ax6.grid()
f6.legend(["Pretrained", "Scratch"])
f6.savefig("FinalEval/f1_vswin")
f6.show()

print("fertsch")