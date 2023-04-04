import hydra
from viskill.trainers.sl_trainer import SkillLearningTrainer


@hydra.main(version_base=None, config_path="./viskill/configs", config_name="skill_learning")
def main(cfg):
    exp = SkillLearningTrainer(cfg)
    exp.train()

if __name__ == "__main__":
    main()