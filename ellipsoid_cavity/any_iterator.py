from multiprocessing import Queue, Process
import tqdm
from dataclasses import dataclass, field
import pickle
import logging

@dataclass
class AnyIteratorConfig:
    ncpus: int=1
    chunksize: int=1
    total: int=None
    desc: str=None
    keep_arg: bool=False
    target_args: tuple=()
    target_kwargs: dict=field(default_factory=dict)
    cache: str = 'salvated.pkl'
    concat: bool = False
    mininterval: float = 0.1
    quiet: bool = False


class AnyIterator(object):
    def __init__(self, target, iterable, post_process_func=None, config=AnyIteratorConfig()):
        self.target = target
        self.processes, self.qin, self.qout = [], None, None
        self.post_process_func = post_process_func
        self.iterable = iterable
        logging.debug(f'config: {config}')
        for key, val in config.__dict__.items():
            self.__setattr__(key, val)
        self.old_ncpus = self.ncpus
        self.total = self.total if self.total is not None else len(iterable)
        logging.debug('attributes moved')

    def initiate_process_pool(self):
        del self.processes
        self.qin, self.qout = Queue(), Queue()
        self.processes = []
        for _ in range(self.ncpus):
            p = Process(target=self.worker)#, (self.qin, self.qout))
            p.daemon = True
            p.start()
            self.processes.append(p)
        logging.debug('processes initiated')

    def result_generator(self):
        for _ in range(self.Nchunks):
            ss, es = self.qout.get()
            for s, e in zip(ss,es):
                yield s,e

    def worker(self):#, qin, qout):
        while True:
            s = self.qin.get()
            if s=='break':
                break
            es = [self.target(x, *self.target_args, **self.target_kwargs) for x in s]
            self.qout.put((s,es))

    def put_chunks(self, data, chunksize):
        chunk = []
        Nchunks =0
        for x in data:
            chunk.append(x)
            if len(chunk)==chunksize:
                self.qin.put(chunk)
                chunk = []
                Nchunks+=1
        if chunk!=[]:
            self.qin.put(chunk)
            Nchunks += 1
        self.Nchunks = Nchunks

    def __enter__(self):
        logging.debug('enter start')
        self.initiate_process_pool()
        self.put_chunks(self.iterable, self.chunksize)
        logging.debug('enter done')
        return self.result_generator()

    def close(self):
        for _ in range(len(self.processes)):
            self.qin.put('break')
        for p in self.processes:
            p.kill()
            p.join()
        self.processes, self.qin, self.qout = [], None, None
        self.ncpus = self.old_ncpus

    def __exit__(self, type, value, traceback):
        logging.debug('exit start')
        self.close()
        logging.debug('exit end')

    def __call__(self):
        result = []
        with self as context:
            for arg, value in tqdm.tqdm(context, total=self.total, desc=self.desc,
                                        mininterval=self.mininterval, disable=self.quiet):
                try:
                    value = value if self.post_process_func is None else self.post_process_func(value)
                    if self.keep_arg:
                        value = (arg, *value) if self.concat else (arg, value)
                    result.append(value)
                except:
                    print('salvating')
                    with open(self.cache, 'wb') as f:
                        pickle.dump(result, f)
                    print('result salvated')
                    raise
        return result

    def consume(self, data_list, chunksize=1):
        if self.processes == []:
            self.initiate_process_pool()

        self.total = len(data_list)
        #ncpus = self.total//self.chunksize + int(self.total%self.chunksize!=0)
        self.iterable = data_list
        self.put_chunks(data_list, chunksize)
            
        generator = self.result_generator()
        if not self.quiet:
            generator = tqdm.tqdm(generator, total=self.total, desc=self.desc,
                                  mininterval=self.mininterval, disable=self.quiet)
            
        return generator

