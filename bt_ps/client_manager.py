import torch
import math
import numpy as np

from math import ceil
from logging import Logger

from p2p_server.utils.utils import ClientSelectionType


class ClientManager:
    def __init__(self, args, logger:Logger, participants_number:int, model_size:int) -> None:
        self.args = args
        self.logger = logger
        self.participants_number = participants_number
        
        # 0 means has not been selected, 1 means has been selected.
        self.selected_history = torch.zeros((participants_number+1), dtype=torch.int)
        self.selected_history[0] = 1
        self.total_arms = []
        
        self.utilities = [0 for _ in range(participants_number+1)]
        self.pwd_round = 0
        # TODO: add round_threshold to cmd line argument
        self.round_threshold = 20
        # self.sample_window = self.args.sample_window
        self.sample_window = 5.0
        self.pacer_step = 20
        self.pacer_delta = 5
        
        # self.epsilon = args.exploration_proportion
        self.exploration_proportion = args.exploration_proportion
        self.last_util_record = 0

        self.exploitClients = []
        self.exploreClients = []
        self.exploitUtilHistory = []
        self.exploreUtilHistory = []
        self.unexplored = set()

        self.model_size= model_size
        self.np_random_state_dict = None

    def init_np_random_state_dict(self):
        if self.np_random_state_dict is not None:
            np.random.set_state(self.np_random_state_dict)

    def init_client_info(self, training_sets):
        # the server
        self.total_arms.append(
            {
                "rank": 0,
                "reward": 0,
                "count": 0,
                "last_involved_round": 0,
                "duration": 1,
                "contribution":0,
            }
        )

        for i in range(self.participants_number):
            # self.logger.info(f"rank {i+1}, training_sets.partitions[i] {len(training_sets.partitions[i])}")
            self.total_arms.append(
                {
                    "rank": i+1,
                    "reward": len(training_sets.partitions[i]), # statistical utility, init by the dataset size like Oort.
                    "count": 0, # times of being selected
                    "last_involved_round": 0, # last round of being selected
                    "duration": 1, # duration of last round, init by 1 like Oort
                    "contribution": 0, # byteswrittendata in P2P
                }
            )

            self.unexplored.add(i+1)

    def update_client_info(self, selection_messages: list):
        """
            use the result of last epoch to update the clients' info.

            1) update reward with statistical_utility.

            2) add selected clients' count in last epoch with 1.
        """
        for selection_message in selection_messages:
            rank = selection_message['rank']
            self.total_arms[rank]['reward'] = selection_message['statistical_utility']
            self.total_arms[rank]['count'] += 1
            self.total_arms[rank]['last_involved_round'] = self.pwd_round
            self.total_arms[rank]['duration'] = selection_message['distribution_time'] + selection_message['calculation_time'] + selection_message['aggregation_time']
            self.total_arms[rank]['contribution'] = int(selection_message['contribution'])

            self.unexplored.discard(rank)

    def calculate_system_utility(self, distribution_time, calculation_time, aggregation_time):
        return 1

    def calculate_utility(self, selection_message: dict):
        """
        utility = loss + distribution_time + calculation_time + aggregation_time

        Args:
            selection_message (dict): {"rank", "statistical_utility", "distribution_time", "calculation_time", "aggregation_time"}
        """
        statistical_utility = selection_message['statistical_utility']
        system_utility = self.calculate_system_utility(
            selection_message['distribution_time'],
            selection_message['calculation_time'],
            selection_message['aggregation_time']
        )
        return statistical_utility * system_utility

    def get_norm(self, aList, clip_bound=0.95, thres=1e-4):
        aList.sort()
        clip_value = aList[min(int(len(aList)*clip_bound), len(aList)-1)]

        _max = aList[-1]
        _min = aList[0]*0.999
        _range = max(_max - _min, thres)
        _avg = sum(aList)/max(1e-4, float(len(aList)))

        return float(_max), float(_min), float(_range), float(_avg), float(clip_value)

    def calculateSumUtil(self, clientList):
        cnt, cntUtil = 1e-4, 0

        for client in clientList:
            if client in self.successfulClients:
                cnt += 1
                cntUtil += self.total_arms[client]['reward']

        return cntUtil/cnt

    def pacer(self):
        # summarize utility in last epoch
        lastExplorationUtil = self.calculateSumUtil(self.exploreClients)
        lastExploitationUtil = self.calculateSumUtil(self.exploitClients)

        self.exploreUtilHistory.append(lastExplorationUtil)
        self.exploitUtilHistory.append(lastExploitationUtil)

        # clear present epoch's successfulClients
        self.successfulClients = set()

        if self.pwd_round >= 2 * self.pacer_step and self.pwd_round % self.pacer_step == 0:
            utilLastPacerRounds = sum(self.exploitUtilHistory[-2*self.pacer_step:-self.pacer_step])
            utilCurrentPacerRounds = sum(self.exploitUtilHistory[-self.pacer_step:])

            # Cumulated statistical utility becomes flat, so we need a bump by relaxing the pacer
            if abs(utilCurrentPacerRounds - utilLastPacerRounds) <= utilLastPacerRounds * 0.1:
                self.round_threshold = min(100., self.round_threshold + self.pacer_delta)
                self.last_util_record = self.pwd_round - self.pacer_step
                self.logger.info("Training selector: Pacer changes at {} to {}".format(self.pwd_round, self.round_threshold))

            # change sharply -> we decrease the pacer step
            elif abs(utilCurrentPacerRounds - utilLastPacerRounds) >= utilLastPacerRounds * 5:
                self.round_threshold = max(self.pacer_delta, self.round_threshold - self.pacer_delta)
                self.last_util_record = self.pwd_round - self.pacer_step
                self.logger.info("Training selector: Pacer changes at {} to {}".format(self.pwd_round, self.round_threshold))

            self.logger.info("Training selector: utilLastPacerRounds {}, utilCurrentPacerRounds {} in round {}"
                .format(utilLastPacerRounds, utilCurrentPacerRounds, self.pwd_round))

        self.logger.info("Training selector: Pacer {}: lastExploitationUtil {}, lastExplorationUtil {}, last_util_record {}".
                        format(self.pwd_round, lastExploitationUtil, lastExplorationUtil, self.last_util_record))

    def select_clients(self, epoch: int, logger, selection_messages: list = None):
        """
            When epoch == 1, selection_messages is None and clients are selected randomly.

            When epoch > 1, selection_messages is not None and clients are selected according to the result of last epoch.

        Args:
            epoch (int): start from 1.
            selection_messages (list, optional): [{"rank", "loss", "distribution_time", "calculation_time", "aggregation_time"}]. Defaults to None.

        Raises:
            ValueError: raised when selection_messages is None and epoch > 1.

        Returns:
            selected_clients: 0/1 tensor. 1 means that the client is selected.
        """
        WORLD_SIZE = self.participants_number + 1
        selected_clients = torch.zeros(torch.Size([WORLD_SIZE]), dtype=torch.uint8)
        # server is always selected
        selected_clients[0] = 1
        selected_clients_number = ceil(self.args.selected_clients_number * self.args.over_commitment) # 1.3K

        self.pwd_round = self.pwd_round+1

        if self.pwd_round == 1 or self.args.client_selection_strategy == ClientSelectionType.RANDOM_STRATEGY.value:
            self.logger.info(f"{self.pwd_round} {self.args.client_selection_strategy} {ClientSelectionType.RANDOM_STRATEGY.value}")
            # in first selection, do random selection.
            selected_clients_rank = torch.randperm(WORLD_SIZE-1).add(1)[:selected_clients_number]
            logger.info(f"{selected_clients_number} {selected_clients_rank}")
            
        elif self.args.client_selection_strategy in [ClientSelectionType.OORT_STRATEGY.value, ClientSelectionType.FedP2P_STRATEGY.value]:
            # update utilities from the result of last epoch
            # for selection_message in selection_messages:
            #     utilities[selection_message['rank']] = self.calculate_utility(selection_message)
            if selection_messages is not None and isinstance(selection_messages, list) and len(selection_messages):
                self.update_client_info(selection_messages)

            self.pacer()
            
            # exploration-exploitation
            # exploration_proportion decays after each round.
            self.exploration_proportion = max(
                self.exploration_proportion*self.args.exploration_proportion_decay, 
                self.args.exploration_proportion_min
            )
            exploration_number = min(
                int(selected_clients_number * self.exploration_proportion),
                torch.nonzero(self.selected_history==0).numel()  # count zero
            )
            exploitation_number = selected_clients_number - exploration_number
            self.logger.info(f"exploration_number: {exploration_number}, exploitation_number: {exploitation_number}")

            # TODO: use pacer to update round_threshold

            # exploitation: select clients according to the result of last epoch
            utilities = self.utilities

            moving_reward, staleness, allloss = [], [], {}
            for item in self.total_arms:
                if item['reward'] > 0:
                    moving_reward.append(item['reward'])
                    staleness.append(self.pwd_round - item['last_involved_round'])

            # update round_prefer_duration
            if self.round_threshold < 100.:
                sortedDuration = sorted([item['duration'] for item in self.total_arms if item['duration']!=1 ])
                logger.info(f"sortedDuration {sortedDuration}")
                self.round_prefer_duration = sortedDuration[
                    min(
                        int(len(sortedDuration) * self.round_threshold/100.), 
                        len(sortedDuration)-1
                    )
                ]
                if self.round_prefer_duration == 0:
                    self.round_prefer_duration = 30.0
            else:
                self.round_prefer_duration = float('inf')
            logger.info(f"round_prefer_duration {self.round_prefer_duration}")
                
            # TODO: add clip_bound to cmd line argument
            # max_reward, min_reward, range_reward, avg_reward, clip_value = self.get_norm(moving_reward, self.args.clip_bound)
            
            max_reward, min_reward, range_reward, avg_reward, clip_value = self.get_norm(moving_reward, 0.9)
            max_staleness, min_staleness, range_staleness, avg_staleness, _ = self.get_norm(staleness, thres=1)

            # calculate utilities
            # scores = [0 for i in range(WORLD_SIZE)]
            scores = {}
            numOfExploited = 0
            for item in self.total_arms:
                if item['rank'] != 0 and item['count'] > 0:
                    numOfExploited += 1
                    # normalize
                    creward = min(item['reward'], clip_value)
                    sc = (creward - min_reward)/float(range_reward)
                    # add temporal uncertainty
                    sc += math.sqrt(0.1*math.log(self.pwd_round)/item['last_involved_round'])

                    # system utility
                    clientDuration = item['duration']
                    if clientDuration > self.round_prefer_duration:
                        system_decay_factor = math.pow(
                            float(self.round_prefer_duration) / max(1e-4, clientDuration),
                            self.args.round_penalty
                        )
                    else:
                        system_decay_factor = 1.0
                    logger.debug(f"sc: {sc}, system_decay_factor: {system_decay_factor} = {sc * system_decay_factor}")
                    sc *= system_decay_factor

                    # contribution
                    if self.args.client_selection_strategy == ClientSelectionType.FedP2P_STRATEGY.value:
                        if self.round_prefer_duration != float("inf"):
                            contribution_decay_factor = math.pow(
                                1.0*item['contribution']/self.model_size,
                                self.args.beta
                            )
                        else:
                            contribution_decay_factor = 1.0
                        # if self.round_prefer_duration == float("inf"):
                        #     contribution_decay_factor = 1.0
                        # else:
                        #     contribution_decay_factor = math.pow(
                        #         1.0*item['contribution']/self.model_size,
                        #         2
                        #     )
                        sc *= contribution_decay_factor
                        if contribution_decay_factor < 1.0:
                            logger.debug(f"punish {item['contribution']} {self.model_size} {contribution_decay_factor}")
                        elif contribution_decay_factor == 1.0:
                            logger.debug(f"stop considering system efficiency")
                        else:
                            logger.debug(f"award {item['contribution']} {self.model_size} {contribution_decay_factor}")

                    if item['last_involved_round']==self.pwd_round:
                        allloss[item['rank']] = sc

                    scores[item['rank']] = abs(sc)

            clientLakes = list(scores.keys())
            # take the top-k, and then sample by probability, take 95% of the cut-off loss
            sortedClientUtil = sorted(scores, key=scores.get, reverse=True)
            # take cut-off utility
            cut_off_util = scores[sortedClientUtil[exploitation_number]] * self.args.cut_off_util

            tempPickedClients = []
            for client_id in sortedClientUtil:
                # we want at least 10 times of clients for augmentation
                if scores[client_id] < cut_off_util and len(tempPickedClients) > 1.5*exploitation_number:
                    break
                tempPickedClients.append(client_id)

            augment_factor = len(tempPickedClients)
            totalSc = max(1e-4, float(sum([scores[key] for key in tempPickedClients])))
            probabilities = [scores[key]/totalSc for key in tempPickedClients]
            exploitation_clients_rank = list(
                np.random.choice(
                    tempPickedClients, 
                    exploitation_number, 
                    p=probabilities, 
                    replace=False
                )
            )
            self.exploitClients = exploitation_clients_rank
            
            exploitation_clients_rank = torch.tensor(exploitation_clients_rank, dtype=torch.long)
            candidate_probabilities = dict(zip(tempPickedClients, probabilities))
            logger.info(f"candidates and probabilities: {candidate_probabilities}, len={len(candidate_probabilities)}, len(tempPickedClients)={len(tempPickedClients)}")
            logger.info(f"exploitation_clients_rank: {exploitation_clients_rank} {exploitation_clients_rank.shape[0]}")
            
            # utilities = scores
            # # utilities = deepcopy(utilities)
            # utilities = torch.tensor(utilities, dtype=torch.float)            
            # # sort and select top K
            # topK_values, exploitation_clients_rank = torch.topk(utilities, exploitation_number)
            # logger.info(f"utilities: {utilities}, topK_values: {topK_values}, exploitation_clients_rank: {exploitation_clients_rank} ")

            # exploration: sample from clients which have not been selected yet.
            # TODO: exploration at present is just random selection.

            # exploration
            _unexplored = [x for x in list(self.unexplored) if x not in self.selected_history.nonzero().flatten().tolist()]
            if len(_unexplored) > 0 and exploration_number > 0:
                init_reward = {}
                for cl in _unexplored:
                    init_reward[cl] = self.total_arms[cl]['reward']
                    clientDuration = self.total_arms[cl]['duration']

                    if clientDuration > self.round_prefer_duration:
                        init_reward[cl] *= ((float(self.round_prefer_duration)/max(1e-4, clientDuration)) ** self.args.round_penalty)

                # prioritize w/ some rewards (i.e., size)
                exploreLen = exploration_number
                pickedUnexploredClients = sorted(init_reward, key=init_reward.get, reverse=True)[:min(int(self.sample_window*exploreLen), len(init_reward))]

                unexploredSc = float(sum([init_reward[key] for key in pickedUnexploredClients]))
                probabilities = [init_reward[key]/max(1e-4, unexploredSc) for key in pickedUnexploredClients]
                candidate_probabilities = dict(zip(pickedUnexploredClients, probabilities))
                logger.info(f"candidates and probabilities: {candidate_probabilities}, len={len(candidate_probabilities)}")

                pickedUnexplored = list(
                    np.random.choice(
                        pickedUnexploredClients, 
                        exploreLen,
                        p=probabilities, replace=False
                    )
                )
                self.exploreClients = pickedUnexplored
            else:
                self.exploreClients = []

            logger.info(f"selected_history: {self.selected_history}, {self.selected_history.sum()}/{self.selected_history.shape[0]}")
            # zero_indices = torch.nonzero(self.selected_history==0).flatten()
            # shuffle_zero_indices = torch.randperm(zero_indices.numel())
            # exploration_clients_rank = zero_indices[shuffle_zero_indices][:exploration_number]
            exploration_clients_rank = torch.tensor(self.exploreClients, dtype=torch.long)
            logger.info(f"exploration_clients_rank: {exploration_clients_rank} {exploration_clients_rank.shape[0]} {self.selected_history[exploration_clients_rank]}")

            selected_clients_rank = torch.cat((exploitation_clients_rank, exploration_clients_rank))

        else:
            raise ValueError

        # logger.info(f"Statistics of selected clients: {self.total_arms}")
        
        # update selected history
        self.selected_history[selected_clients_rank] = 1

        selected_clients[selected_clients_rank] = 1

        self.np_random_state_dict = np.random.get_state()

        return selected_clients
