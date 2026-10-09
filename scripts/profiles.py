"""Explicit speed and lookahead contracts; weights never migrate implicitly."""
SPEED_516 = {'id': 'speed-516', 'version': 2, 'architecture': [5, 16, 1],
             'inputNames': ['birdY', 'velocityY', 'pipeDistance', 'gapY', 'pipeSpeed']}
LOOKAHEAD_616 = {'id': 'lookahead-616', 'version': 3, 'architecture': [6, 16, 1],
                 'inputNames': SPEED_516['inputNames'] + ['followingGapY']}

def profile_for_id(profile_id):
    for profile in (SPEED_516, LOOKAHEAD_616):
        if profile['id'] == profile_id:
            return profile
    raise ValueError('Unsupported model profile.')

def profile_for_architecture(architecture):
    if architecture == SPEED_516['architecture']:
        return SPEED_516
    if architecture == LOOKAHEAD_616['architecture']:
        return LOOKAHEAD_616
    raise ValueError('Supported architectures are 5 -> 16 -> 1 and 6 -> 16 -> 1.')
