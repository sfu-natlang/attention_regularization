import torch

import onmt
from onmt.bin.train import _get_parser
from onmt.train_single import configure_process, _tally_parameters
from onmt.utils.logging import init_logger, logger
from onmt.utils.parse import ArgumentParser
from onmt.inputters.inputter import build_dataset_iter, \
    load_old_vocab, old_style_vocab, build_dataset_iter_multiple
from onmt.model_builder import build_model


def main():
    parser = _get_parser()

    opt = parser.parse_args()
    validate(opt)


def validate(opt, device_id=0):
    configure_process(opt, device_id)
    configure_process
    if opt.train_from:
        logger.info('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)
        model_opt = ArgumentParser.ckpt_model_opts(checkpoint["opt"])
        ArgumentParser.update_model_opts(model_opt)
        ArgumentParser.validate_model_opts(model_opt)
        logger.info('Loading vocab from checkpoint at %s.' % opt.train_from)
        vocab = checkpoint['vocab']

        if old_style_vocab(vocab):
            fields = load_old_vocab(
                vocab, opt.model_type, dynamic_dict=opt.copy_attn)
        else:
            fields = vocab

    # Report src and tgt vocab sizes, including for features
    for side in ['src', 'tgt']:
        f = fields[side]
        try:
            f_iter = iter(f)
        except TypeError:
            f_iter = [(side, f)]
        for sn, sf in f_iter:
            if sf.use_vocab:
                logger.info(' * %s vocab size = %d' % (sn, len(sf.vocab)))

    
    model = build_model(model_opt, opt, fields, checkpoint)
    n_params, enc, dec = _tally_parameters(model)
    logger.info('encoder: %d' % enc)
    logger.info('decoder: %d' % dec)
    logger.info('* number of parameters: %d' % n_params)
    #_check_save_model_path(opt)
    
    valid_iter = build_dataset_iter("valid", fields, opt, is_train=False)

    tgt_field = dict(fields)["tgt"].base_field
    valid_loss = onmt.utils.loss.build_loss_compute(
        model, tgt_field, opt, train=False)

    model.eval()

    with torch.no_grad():
        stats = onmt.utils.Statistics()

        for batch in valid_iter:
            
            src, src_lengths = batch.src if isinstance(batch.src, tuple) \
                                   else (batch.src, None)
            tgt = batch.tgt

            # F-prop through the model.
            outputs, attns = model(src, tgt, src_lengths)

            # Compute loss.
            _, batch_stats = valid_loss(batch, outputs, attns)

            # Update statistics.
            stats.update(batch_stats)

    print('n words:  %d' % stats.n_words)
    print('Validation perplexity: %g' % stats.ppl())
    print('Validation accuracy: %g' % stats.accuracy())
    print('Validation avg attention entropy: %g' % stats.attn_entropy())

    

if __name__ == "__main__":
    main()