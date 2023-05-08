import hydra
from viskill.trainers.sc_trainer import SkillChainingTrainer


@hydra.main(version_base=None, config_path="./viskill/configs", config_name="eval")
def main(cfg):
    exp = SkillChainingTrainer(cfg)
    exp.eval_ckpt()

if __name__ == "__main__":
    main()