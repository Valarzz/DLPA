from mpc_model.hard_code_utils import *


class worldmodel:
    def __init__(self, label):
        # Entities
        self.xpos = 0.0
        self.platform1 = Platform(0.0, Constants.HEIGHT1, Constants.WIDTH1)
        self.platform2 = Platform(Constants.GAP1 + self.platform1.size[0], Constants.HEIGHT2, Constants.WIDTH2)
        self.platform3 = Platform(self.platform2.position[0] +
                                  Constants.GAP2 + self.platform2.size[0], Constants.HEIGHT3, Constants.WIDTH3)
        self.label = label
        self.mode = "hard"

    def predict(self, state, action):
        terminal = False
        running = True

        # act_index = action[0]
        # act = ACTION_LOOKUP[act_index]
        # param = (action[1][action[0]]+1)/2. * Constants.PARAMETERS_MAX[action[0]]
        # param = np.clip(param, Constants.PARAMETERS_MIN[act_index], Constants.PARAMETERS_MAX[act_index])

        act_index = action[:3].argmax()
        act = ACTION_LOOKUP[act_index]
        param = (action[-1]+1)/2. * Constants.PARAMETERS_MAX[act_index]
        param = np.clip(param, Constants.PARAMETERS_MIN[act_index], Constants.PARAMETERS_MAX[act_index])

        steps = 0
        difft = 1.0
        reward = 0.

        rescale_state = (state + 1) / 2 * Constants.SCALE_VECTOR - Constants.SHIFT_VECTOR
        ppos, pvol = rescale_state[0], rescale_state[1]
        epos, evol = rescale_state[2], rescale_state[3]

        self.player = Player(ppos, pvol)
        if self.player.position[0] > self.platform2.position[0]:
            eplatform = self.platform2
        else:
            eplatform = self.platform1
        self.enemy = Enemy(eplatform, epos, evol)

        self.xpos = self.player.position[0]

        # update until back on platform
        while running:
            reward, terminal = self._update(act, param)

            if act == RUN:
                difft -= Constants.DT
                running = difft > 0
            elif act in [JUMP, HOP, LEAP]:
                running = not self._on_platforms()

            if terminal:
                running = False
            steps += 1

        try:
            reward = reward[0]
        except:
            pass

        obs = self.get_state()
        for i in range(len(obs)):
            try:
                obs[i] = obs[i][0] * 2 - 1
            except:
                obs[i] = obs[i] * 2 - 1

        if self.label == "state":
            return obs
        elif self.label == "reward":
            return reward
        elif self.label == "terminal":
            return terminal
        elif self.label == "all":
            return obs, reward, terminal
        else:
            raise "bad"

    def _update(self, act, param, dt=Constants.DT):
        self._perform_action(act, param, dt)
        if self._on_platforms():
            self.player.ground_bound()
        for entity in [self.player, self.enemy]:
            entity.update(dt)
        for platform in [self.platform1, self.platform2, self.platform3]:
            if self.player.colliding(platform):
                self.player.decollide(platform)
                self.player.velocity[0] = 0.0
        reward = (self.player.position[0] - self.xpos) / self._right_bound()
        return self._terminal_check(reward)

    def _perform_action(self, act, parameters, dt=Constants.DT):

        if self._on_platforms():
            if act == JUMP:
                self.player.jump(parameters)
            elif act == RUN:
                self.player.run(parameters, dt)
            elif act == LEAP:
                self.player.leap_to(parameters)
            elif act == HOP:
                self.player.hop_to(parameters)
        else:
            self.player.fall()

    def _on_platforms(self):
        for platform in [self.platform1, self.platform2, self.platform3]:
            if self.player.on_platform(platform):
                return True
        return False

    def _lower_bound(self):
        """ Returns the lowest height of the platforms. """
        lower = min(self.platform1.position[1], self.platform2.position[1], self.platform3.position[1])
        return lower

    def _right_bound(self):
        return self.platform3.position[0] + self.platform3.size[0]

    def _terminal_check(self, reward=0.0):
        """ Determines if the episode is ended, and the reward. """
        end_episode = self.player.position[1] < self._lower_bound() + Constants.PLATFORM_HEIGHT
        right = self.player.position[0] >= self._right_bound()
        for entity in [self.enemy]:
            if self.player.colliding(entity):
                end_episode = True
        if right:
            reward = (self._right_bound() - self.xpos) / self._right_bound()
            end_episode = True
        return reward, end_episode

    def get_state(self):
        """ Returns the scaled representation of the current state. """
        basic_features = [
            self.player.position[0],  # 0
            self.player.velocity[0],  # 1
            self.enemy.position[0],  # 2
            self.enemy.dx]  # 3
        platform_features = self._platform_features(basic_features)
        state = np.concatenate((basic_features, platform_features))
        scaled_state = self._scale_state(state)
        return scaled_state

    def _platform_features(self, basic_features):
        """
        Compute the implicit features of the platforms.

        Parameters
        ----------
        basic_features :
        """
        xpos = basic_features[0]
        if xpos < Constants.WIDTH1 + Constants.GAP1:
            pos = 0.0
            wd1 = Constants.WIDTH1
            wd2 = Constants.WIDTH2
            gap = Constants.GAP1
            diff = Constants.HEIGHT2 - Constants.HEIGHT1
        elif xpos < Constants.WIDTH1 + Constants.GAP1 + Constants.WIDTH2 + Constants.GAP2:
            pos = Constants.WIDTH1 + Constants.GAP1
            wd1 = Constants.WIDTH2
            wd2 = Constants.WIDTH3
            gap = Constants.GAP2
            diff = Constants.HEIGHT3 - Constants.HEIGHT2
        else:
            pos = Constants.WIDTH1 + Constants.GAP1 + Constants.WIDTH2 + Constants.GAP2
            wd1 = Constants.WIDTH3
            wd2 = 0.0
            gap = 0.0
            diff = 0.0
        return [wd1, wd2, gap, pos, diff]

    def _scale_state(self, state):
        scaled = (state + Constants.SHIFT_VECTOR) / Constants.SCALE_VECTOR
        return scaled





